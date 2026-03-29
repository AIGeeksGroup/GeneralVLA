#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上位机 TCP 客户端。运行在本机，连接下位机 10.5.23.176：
  - 发送截图指令，接收彩图与深度图
  - 发送 pose（x,y,z,rx,ry,rz,grip）给下位机

当前协议约定：
  - x,y,z 为**相机坐标系下**的末端位置（单位 mm，z 轴沿相机光轴）
  - rx,ry,rz,grip 按原有语义透传给下位机/机械臂使用（单位 deg / 夹爪开合）

用法:
  python3 upper_client.py                    # 交互：输入 capture / pose 或带参数
  python3 upper_client.py capture            # 请求一帧并保存/显示
  python3 upper_client.py pose 138 -2.74 160.77 -171.41 -1.31 -82.54 0   # 发送位姿（相机坐标）
"""
import argparse
import struct
import socket
import sys
import os
from datetime import datetime

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# --------------- 协议（与 lower_server 一致）---------------
# 上行: "CAPTURE\n" 或
#       "POSE x y z rx ry rz grip\n"
#       其中 x,y,z 为**相机坐标系**下的位置（mm），rx,ry,rz 为角度（deg），grip 为夹爪指令
# 下行 CAPTURE: "OK\n"
#              + 9*8B K(3x3, float64, 行优先)
#              + 4B color_len(LE) + color_jpeg
#              + 4B H + 4B W + H*W*4 depth_float32
# 下行 POSE: "OK\n" 或 "ERR ...\n"


def recv_exact(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError("连接关闭")
        buf += chunk
    return buf


def recv_line(sock) -> str:
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("连接关闭")
        buf += chunk
    line, _ = buf.split(b"\n", 1)
    return line.decode("utf-8", errors="replace").strip()


class UpperClient:
    def __init__(self, host: str = "10.5.23.176", port: int = 8888, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock = None

    def connect(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))
        print(f"已连接 {self.host}:{self.port}")

    def close(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _send_cmd(self, cmd: str) -> str:
        if self._sock is None:
            self.connect()
        self._sock.sendall((cmd.strip() + "\n").encode("utf-8"))
        return recv_line(self._sock)

    def capture(self):
        """请求一帧彩图+深度图，返回 (color_bgr, depth_float32)，单位米。"""
        resp = self._send_cmd("CAPTURE")
        if resp.startswith("ERR"):
            raise RuntimeError(resp)
        if resp != "OK":
            raise RuntimeError(f"意外响应: {resp}")
        # 先收 3x3 相机内参 K（float64，行优先）
        K_bytes = recv_exact(self._sock, 9 * 8)
        K = np.frombuffer(K_bytes, dtype=np.float64).reshape(3, 3)
        color_len = struct.unpack("<I", recv_exact(self._sock, 4))[0]
        color_bytes = recv_exact(self._sock, color_len)
        h, w = struct.unpack("<II", recv_exact(self._sock, 8))
        depth_len = h * w * 4
        depth_bytes = recv_exact(self._sock, depth_len)
        depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(h, w)
        color_bgr = cv2.imdecode(np.frombuffer(color_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise RuntimeError("彩图解码失败")
        return color_bgr, depth, K

    def send_pose(self, x: float, y: float, z: float, rx: float, ry: float, rz: float, grip: float):
        """
        发送位姿给下位机（单位：mm, deg）。

        约定：
        - x,y,z 为相机坐标系下的位置（mm）
        - rx,ry,rz 为角度（deg），语义由下位机/机械臂侧决定（透传）
        """
        cmd = f"POSE {x} {y} {z} {rx} {ry} {rz} {grip}"
        resp = self._send_cmd(cmd)
        if resp.startswith("ERR"):
            raise RuntimeError(resp)
        if resp != "OK":
            raise RuntimeError(f"意外响应: {resp}")
        print("位姿已发送并执行")


def load_txt_matrix(path, shape):
    arr = np.loadtxt(path, dtype=np.float64)
    return np.asarray(arr, dtype=np.float64).reshape(shape)


def transform_cam_to_base(p_cam, R, t):
    return R @ p_cam + t


class _ClickState:
    def __init__(
        self,
        color_bgr,
        depth_m,
        fx,
        fy,
        cx,
        cy,
        depth_radius=4,
        min_depth=0.25,
        max_depth=3.0,
        window_name="Click Color",
        depth_window_name="Click Depth",
    ):
        self.color = color_bgr
        self.depth = depth_m
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_radius = depth_radius
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.window_name = window_name
        self.depth_window_name = depth_window_name

        self.clicked_uv = None
        self.clicked_cam_point = None
        self.clicked_depth_val = None
        self.status_text = "Click on target point"

        cv2.namedWindow(self.window_name)
        cv2.namedWindow(self.depth_window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    @staticmethod
    def _get_valid_depth_median(depth_img, u, v, radius=4, min_d=0.25, max_d=3.0):
        h, w = depth_img.shape
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))
        vals = []
        for yy in range(max(0, v - radius), min(h, v + radius + 1)):
            for xx in range(max(0, u - radius), min(w, u + radius + 1)):
                d = depth_img[yy, xx]
                if np.isfinite(d) and min_d < d < max_d:
                    vals.append(d)
        return float(np.median(vals)) if len(vals) > 0 else 0.0

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.color is None or self.depth is None or self.fx is None:
            self.status_text = "Camera data not ready"
            return

        depth = self.depth
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        depth_val = self._get_valid_depth_median(
            depth, x, y,
            radius=self.depth_radius,
            min_d=self.min_depth,
            max_d=self.max_depth,
        )

        self.clicked_uv = (x, y)
        if depth_val <= 0:
            self.clicked_cam_point = None
            self.clicked_depth_val = None
            self.status_text = "Depth invalid"
            return

        X = (x - cx) * depth_val / fx
        Y = (y - cy) * depth_val / fy
        Z = depth_val

        self.clicked_cam_point = np.array([X, Y, Z], dtype=np.float64)
        self.clicked_depth_val = depth_val
        self.status_text = "Point locked"
        print(f"[已锁定相机点] Pixel=({x},{y}) Cam XYZ={np.round(self.clicked_cam_point, 6)}")

    def reset(self):
        self.clicked_uv = None
        self.clicked_cam_point = None
        self.clicked_depth_val = None
        self.status_text = "Click on target point"

    def _make_depth_vis(self):
        depth = np.nan_to_num(self.depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_norm = (depth - self.min_depth) / (self.max_depth - self.min_depth)
        depth_norm = np.clip(depth_norm, 0, 1)
        depth_vis = (depth_norm * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        return depth_vis

    def get_visuals(self):
        color = self.color.copy()
        depth_vis = self._make_depth_vis()

        if self.clicked_uv is None:
            cv2.putText(color, self.status_text, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return None, color, depth_vis

        u, v = self.clicked_uv
        cv2.circle(color, (u, v), 6, (0, 255, 0), -1)
        cv2.circle(depth_vis, (u, v), 6, (0, 255, 0), -1)

        if self.clicked_cam_point is None or self.clicked_depth_val is None:
            cv2.putText(color, self.status_text, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return None, color, depth_vis

        txt0 = f"Status: {self.status_text}"
        txt1 = f"Pixel(u,v)=({u},{v}) Locked Depth={self.clicked_depth_val:.4f}m"
        txt2 = (
            f"Locked Cam XYZ(m): "
            f"[{self.clicked_cam_point[0]:.4f}, {self.clicked_cam_point[1]:.4f}, {self.clicked_cam_point[2]:.4f}]"
        )
        cv2.putText(color, txt0, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(color, txt1, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(color, txt2, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return self.clicked_cam_point.copy(), color, depth_vis


def run_click_mode(
    client: UpperClient,
    R_path: str,
    t_path: str,
    hover_mm: float = 5.0,
    depth_radius: int = 4,
    min_depth: float = 0.25,
    max_depth: float = 3.0,
    move_vec: int = 30,
):
    if cv2 is None:
        print("需要 opencv-python: pip install opencv-python", file=sys.stderr)
        sys.exit(1)

    # R = load_txt_matrix(R_path, (3, 3))
    # t = load_txt_matrix(t_path, (3,)).reshape(3)

    # print("\n========== 已加载手眼标定 ==========")
    # print("注意：当前版本仅在上位机中使用手眼标定参数做参考，真正的坐标系转换在下位机完成。")
    # print("参考变换: p_base = R @ p_cam + t")
    # print("R =\n", R)
    # print("t =", t)

    # fixed_rpy_deg 为末端在「相机坐标系」下的期望姿态 (rx, ry, rz, 单位 deg)
    # 由原先 base 坐标系下的姿态 (-171.41, -1.31, -82.54) 结合手眼标定 R 反算得到
    fixed_rpy_deg = (7.71, 1.86, -98.87)
    fixed_grip = 0.0
    # init_pose 的位置部分改为相机坐标系下的值（由原 base 坐标通过手眼标定 R,t 反算得到）
    # 原 base 坐标(mm): [138.26, -2.74, 160.77]
    # 反算得到 cam 坐标(mm): [28.12, -231.47, 371.47]
    # 姿态部分使用 fixed_rpy_deg（相机坐标系下）
    init_pose = [28.12, -231.47, 371.47, 7.71, 1.86, -98.87, 0.0]

    print("\n=== 操作说明 ===")
    print("鼠标左键: 点击图像中的目标点，锁定相机三维点")
    print("回车键: 计算位姿并发送给下位机 (末端在该点上方 hover_mm)")
    print("空格键: 回到固定初始位姿")
    print("c: 清除当前点击点")
    print("n: 向下位机请求新的一帧图像")
    print("q: 退出 click 模式")

    # 截图保存目录
    save_root = os.path.join(os.getcwd(), "click_frames")
    os.makedirs(save_root, exist_ok=True)
    frame_idx = 0

    try:
        while True:
            print("从下位机请求一帧彩图+深度图+内参 K...")
            color, depth, K = client.capture()
            print(f"收到彩图 {color.shape[1]}x{color.shape[0]} 深度 {depth.shape[1]}x{depth.shape[0]} (float32 米)")
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            # 保存当前帧到专用文件夹
            frame_idx += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            prefix = f"frame_{frame_idx:04d}_{ts}"
            color_path = os.path.join(save_root, f"{prefix}_color.jpg")
            depth_npy_path = os.path.join(save_root, f"{prefix}_depth.npy")
            depth_vis_path = os.path.join(save_root, f"{prefix}_depth_vis.png")
            np.save(depth_npy_path, depth)
            depth_vis = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            valid = (depth_vis > 0) & (depth_vis < 5)
            if valid.any():
                dmin, dmax = depth_vis[valid].min(), depth_vis[valid].max()
            else:
                dmin, dmax = 0.0, 5.0
            depth_vis = np.clip((depth_vis - dmin) / (dmax - dmin + 1e-6), 0, 1)
            depth_vis = (depth_vis * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imwrite(color_path, color)
            cv2.imwrite(depth_vis_path, depth_vis)
            print(f"[保存当前帧] {os.path.relpath(color_path)} , {os.path.relpath(depth_vis_path)} , {os.path.relpath(depth_npy_path)}")

            click_state = _ClickState(
                color_bgr=color,
                depth_m=depth,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                depth_radius=depth_radius,
                min_depth=min_depth,
                max_depth=max_depth,
                window_name="TCP Click Color",
                depth_window_name="TCP Click Depth",
            )

            last_target_cam = None

            while True:
                p_cam, vis, depth_vis = click_state.get_visuals()

                if p_cam is not None:
                    # 记录最新的相机坐标点（用于真正发送给下位机）
                    last_target_cam = p_cam.copy()

                if vis is not None:
                    if last_target_cam is not None:
                        txt3 = (
                            f"Locked Cam XYZ(m): "
                            f"[{last_target_cam[0]:.4f}, {last_target_cam[1]:.4f}, {last_target_cam[2]:.4f}]"
                        )
                        txt4 = (
                            f"Send Cam(mm): "
                            f"[{last_target_cam[0]*1000:.1f}, "
                            f"{last_target_cam[1]*1000:.1f}, "
                            f"{last_target_cam[2]*1000 + hover_mm:.1f}]"
                        )
                        cv2.putText(vis, txt3, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
                        cv2.putText(vis, txt4, (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

                    cv2.imshow(click_state.window_name, vis)

                if depth_vis is not None:
                    cv2.imshow(click_state.depth_window_name, depth_vis)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("退出 click 模式。")
                    return
                elif key == 32:  # 空格键: 回到初始位姿
                    print("[回到写死初始位姿]")
                    print(
                        f"XYZ(mm)=({init_pose[0]:.2f}, {init_pose[1]:.2f}, {init_pose[2]:.2f}), "
                        f"RPY(deg)=({init_pose[3]:.2f}, {init_pose[4]:.2f}, {init_pose[5]:.2f}), "
                        f"grip={init_pose[6]:.2f}"
                    )
                    client.send_pose(
                        init_pose[0],
                        init_pose[1],
                        init_pose[2],
                        init_pose[3],
                        init_pose[4],
                        init_pose[5],
                        init_pose[6],
                    )
                elif key == ord("c"):
                    click_state.reset()
                    last_target_cam = None
                    print("已清除当前点击点。")
                elif key == ord("n"):
                    print("请求新的一帧图像。")
                    break  # 跳出内层循环，重新 CAPTURE 一帧
                elif key == 13:
                    if p_cam is None or last_target_cam is None:
                        print("[跳过] 请先在图像中点击一个有效点。")
                        continue

                    # 协议要求：发送相机坐标系下的位置（mm）
                    target_x_mm = last_target_cam[0] * 1000.0
                    target_y_mm = last_target_cam[1] * 1000.0
                    target_z_mm = last_target_cam[2] * 1000.0 + hover_mm

                    print("[发送位姿到下位机]")
                    print(f"Cam XYZ(m): {np.round(p_cam, 6)}")
                    print(f"Send Cam XYZ(mm): ({target_x_mm:.2f}, {target_y_mm:.2f}, {target_z_mm:.2f})")
                    print(
                        f"Fixed RPY(deg): ({fixed_rpy_deg[0]:.2f}, "
                        f"{fixed_rpy_deg[1]:.2f}, {fixed_rpy_deg[2]:.2f}) grip={fixed_grip:.2f}"
                    )

                    client.send_pose(
                        target_x_mm,
                        target_y_mm,
                        target_z_mm,
                        fixed_rpy_deg[0],
                        fixed_rpy_deg[1],
                        fixed_rpy_deg[2],
                        fixed_grip,
                    )
                    # 不退出整个程序，继续在当前这一帧上允许继续点击/发送
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="上位机 TCP 客户端：截图 + 发送机械臂位姿")
    parser.add_argument("--host", default="10.5.23.176", help="下位机 IP")
    parser.add_argument("--port", type=int, default=8888, help="下位机端口")
    parser.add_argument("command", nargs="?", help="capture=截图, pose=发位姿, click=点击计算位姿；不填则进入交互")
    parser.add_argument("args", nargs="*", help="pose 时: x y z rx ry rz grip；click 时: R_path t_path [hover_mm]")
    a = parser.parse_args()

    client = UpperClient(host=a.host, port=a.port)
    try:
        if a.command and a.command.lower() == "capture":
            if not cv2:
                print("需要 opencv-python: pip install opencv-python", file=sys.stderr)
                sys.exit(1)
            color, depth = client.capture()
            print(f"收到彩图 {color.shape[1]}x{color.shape[0]} 深度 {depth.shape[1]}x{depth.shape[0]} (float32 米)")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            base = os.getcwd()
            color_path = os.path.join(base, f"color_{ts}.jpg")
            depth_npy_path = os.path.join(base, f"depth_{ts}.npy")
            # 保存原始深度
            np.save(depth_npy_path, depth)
            # 深度可视化
            depth_vis = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            valid = (depth_vis > 0) & (depth_vis < 5)
            if valid.any():
                dmin, dmax = depth_vis[valid].min(), depth_vis[valid].max()
            else:
                dmin, dmax = 0.0, 5.0
            depth_vis = np.clip((depth_vis - dmin) / (dmax - dmin + 1e-6), 0, 1)
            depth_vis = (depth_vis * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            depth_png_path = os.path.join(base, f"depth_vis_{ts}.png")
            cv2.imwrite(color_path, color)
            cv2.imwrite(depth_png_path, depth_vis)
            print(
                "已保存: "
                f"{os.path.basename(color_path)}, "
                f"{os.path.basename(depth_png_path)}, "
                f"{os.path.basename(depth_npy_path)}"
            )

        elif a.command and a.command.lower() == "pose":
            if len(a.args) != 7:
                print("pose 需要 7 个参数: x y z rx ry rz grip (单位 mm, deg)", file=sys.stderr)
                sys.exit(1)
            vals = [float(x) for x in a.args]
            client.send_pose(*vals)

        elif a.command and a.command.lower() == "click":
            if not cv2:
                print("需要 opencv-python: pip install opencv-python", file=sys.stderr)
                sys.exit(1)
            if len(a.args) < 2:
                print(
                    "click 需要参数: R_path t_path [hover_mm]",
                    file=sys.stderr,
                )
                sys.exit(1)
            R_path = a.args[0]
            t_path = a.args[1]
            hover_mm = float(a.args[2]) if len(a.args) >= 3 else 5.0
            run_click_mode(
                client,
                R_path=R_path,
                t_path=t_path,
                hover_mm=hover_mm,
            )

        else:
            # 交互模式：每次由用户决定 capture 还是 pose
            print("命令: capture | pose x y z rx ry rz grip | quit")
            while True:
                line = input("> ").strip()
                if not line or line.lower() == "quit":
                    break
                parts = line.split()
                if parts[0].lower() == "capture":
                    try:
                        color, depth = client.capture()
                        print(f"收到彩图 {color.shape[1]}x{color.shape[0]} 深度 {depth.shape[1]}x{depth.shape[0]}")
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        base = os.getcwd()
                        color_path = os.path.join(base, f"color_{ts}.jpg")
                        depth_npy_path = os.path.join(base, f"depth_{ts}.npy")
                        # 保存原始深度
                        np.save(depth_npy_path, depth)
                        # 深度可视化
                        depth_vis = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                        valid = (depth_vis > 0) & (depth_vis < 5)
                        if valid.any():
                            dmin, dmax = depth_vis[valid].min(), depth_vis[valid].max()
                        else:
                            dmin, dmax = 0.0, 5.0
                        depth_vis = np.clip((depth_vis - dmin) / (dmax - dmin + 1e-6), 0, 1)
                        depth_vis = (depth_vis * 255).astype(np.uint8)
                        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                        depth_png_path = os.path.join(base, f"depth_vis_{ts}.png")
                        cv2.imwrite(color_path, color)
                        cv2.imwrite(depth_png_path, depth_vis)
                        print(
                            "已保存: "
                            f"{os.path.basename(color_path)}, "
                            f"{os.path.basename(depth_png_path)}, "
                            f"{os.path.basename(depth_npy_path)}"
                        )
                    except Exception as e:
                        print("错误:", e)
                elif parts[0].lower() == "pose" and len(parts) == 8:
                    try:
                        client.send_pose(*[float(x) for x in parts[1:8]])
                    except Exception as e:
                        print("错误:", e)
                else:
                    print("未知命令或参数不足")
    finally:
        client.close()


if __name__ == "__main__":
    main()
