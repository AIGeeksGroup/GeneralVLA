#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下位机 TCP 服务端。部署在 10.5.23.176，通过 TCP 接收上位机指令：
  - 截图指令：从 ROS2 /camera/color/image_raw 和 /camera/aligned_depth_to_color/image_raw 取彩图与深度图，回传上位机
  - 位姿指令：接收**相机坐标系下**的 pose，将其转换到机械臂 base 坐标系后，发送给机械臂 (PiperX_Arm)

运行方式（在下位机 10.5.23.176 上）:
  ssh common@10.5.23.176
  cd /home/common/eye_hand   # 或包含 piperx_arm_template 的目录
  source /opt/ros/jazzy/setup.bash
  python3 lower_server.py [--host 0.0.0.0] [--port 8888] --R-path R.txt --t-path t.txt
"""
import sys
import os
import struct
import threading
import argparse
import socket
import numpy as np

# 下位机 ROS2 环境
sys.path.insert(0, "/opt/ros/jazzy/lib/python3.12/site-packages")
sys.path.insert(0, os.path.expanduser("~/eye_hand") if os.path.exists(os.path.expanduser("~/eye_hand")) else "")
sys.path.insert(0, "/home/common/eye_hand")

# 避免 GUI 相关环境导致问题
for k in ("QT_AUTO_SCREEN_SCALE_FACTOR", "QT_SCALE_FACTOR", "QT_FONT_DPI", "SESSION_MANAGER"):
    os.environ.pop(k, None)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters

# 机械臂接口（与 click_to_point_above_move_fixed_init 一致）
try:
    from piperx_arm_template import PiperX_Arm
except ImportError:
    PiperX_Arm = None


# --------------- 协议约定 ---------------
# 上行（上位机 -> 下位机）: 一行文本，以 \n 结尾
#   "CAPTURE"     -> 请求一帧彩图+深度图
#   "POSE x y z rx ry rz grip"
#       - x,y,z：末端在相机坐标系下的位置（mm）
#       - rx,ry,rz：末端在相机坐标系下的姿态 (roll, pitch, yaw, 单位 deg)
#       - grip：夹爪指令
# 下行（下位机 -> 上位机）:
#   CAPTURE 成功: "OK\n"
#                 + 9*8B K(3x3, float64, 行优先，LE)
#                 + 4B color_len(LE) + color_jpeg
#                 + 4B H(LE) + 4B W(LE) + H*W*4 depth_float32
#   CAPTURE 失败: "ERR message\n"
#   POSE 成功: "OK\n"
#   POSE 失败: "ERR message\n"


def _load_txt_matrix(path, shape):
    """从 txt 加载矩阵/向量，shape 为期望形状。"""
    arr = np.loadtxt(path, dtype=np.float64)
    return np.asarray(arr, dtype=np.float64).reshape(shape)


def _euler_zyx_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    将 (rx, ry, rz) roll-pitch-yaw 角（弧度）转换为旋转矩阵，约定:
      R = Rz(yaw=rz) @ Ry(pitch=ry) @ Rx(roll=rx)
    """
    cr, sr = np.cos(rx), np.sin(rx)
    cp, sp = np.cos(ry), np.sin(ry)
    cy, sy = np.cos(rz), np.sin(rz)
    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    Ry = np.array([[cp, 0.0, sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0, cp]], dtype=np.float64)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cr, -sr],
                   [0.0, sr,  cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _matrix_to_euler_zyx(Rm: np.ndarray) -> tuple[float, float, float]:
    """
    将旋转矩阵转换为 roll-pitch-yaw (rx, ry, rz) 弧度，约定:
      R = Rz(rz) @ Ry(ry) @ Rx(rx)
    """
    # 防数值误差
    r20 = float(Rm[2, 0])
    if abs(r20) < 1.0 - 1e-8:
        pitch = -np.arcsin(r20)
        roll = np.arctan2(Rm[2, 1], Rm[2, 2])
        yaw = np.arctan2(Rm[1, 0], Rm[0, 0])
    else:
        # 万向节锁，设 yaw = 0
        yaw = 0.0
        if r20 <= -1.0:
            pitch = np.pi / 2.0
            roll = np.arctan2(Rm[0, 1], Rm[0, 2])
        else:
            pitch = -np.pi / 2.0
            roll = np.arctan2(-Rm[0, 1], -Rm[0, 2])
    return roll, pitch, yaw


class RGBDSyncNode(Node):
    """订阅彩图+深度图+CameraInfo 并保持最新一帧（带同步）。"""
    def __init__(self, color_topic: str, depth_topic: str, info_topic: str = "/camera/color/camera_info"):
        super().__init__("rgbd_sync_node")
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._color_bgr = None
        self._depth_m = None  # float32, 单位米
        self._K = None        # 3x3 相机内参

        # 与 click_to_point_above_move_fixed_init 一致，不指定 qos 以兼容下位机环境
        color_sub = message_filters.Subscriber(self, Image, color_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self._ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=5, slop=0.05
        )
        self._ts.registerCallback(self._cb)

        # CameraInfo 订阅
        self._sub_info = self.create_subscription(
            CameraInfo, info_topic, self._info_cb, 10
        )

    def _cb(self, color_msg, depth_msg):
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            if depth_msg.encoding == "16UC1":
                depth = depth.astype(np.float32) / 1000.0
            else:
                depth = depth.astype(np.float32)
            with self._lock:
                self._color_bgr = color
                self._depth_m = depth
        except Exception as e:
            self.get_logger().warn(f"image cb error: {e}")

    def _info_cb(self, msg: CameraInfo):
        # 提取 3x3 K 矩阵
        K = np.array(
            [
                [msg.k[0], msg.k[1], msg.k[2]],
                [msg.k[3], msg.k[4], msg.k[5]],
                [msg.k[6], msg.k[7], msg.k[8]],
            ],
            dtype=np.float64,
        )
        with self._lock:
            self._K = K

    def get_latest(self):
        with self._lock:
            c = self._color_bgr.copy() if self._color_bgr is not None else None
            d = self._depth_m.copy() if self._depth_m is not None else None
            K = self._K.copy() if self._K is not None else None
        return c, d, K


def run_tcp_server(host: str, port: int, rgbd_node: RGBDSyncNode, robot,
                   R_cam2base: np.ndarray | None = None,
                   t_cam2base: np.ndarray | None = None):
    import cv2
    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_sock.bind((host, port))
    listen_sock.listen(2)
    listen_sock.settimeout(1.0)
    print(f"[TCP] 监听 {host}:{port}，等待上位机连接...")

    while True:
        try:
            conn, addr = listen_sock.accept()
        except socket.timeout:
            continue
        conn.settimeout(30.0)
        print(f"[TCP] 连接来自 {addr}")
        try:
            buf = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    line = buf.split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
                    buf = buf[buf.index(b"\n") + 1:]
                    if not line:
                        continue
                    parts = line.split()
                    cmd = parts[0].upper() if parts else ""

                    if cmd == "CAPTURE":
                        color, depth, K = rgbd_node.get_latest()
                        if color is None or depth is None:
                            conn.sendall(b"ERR no image\n")
                            continue
                        if K is None:
                            conn.sendall(b"ERR no camera_info\n")
                            continue
                        # 彩图 JPEG 编码
                        _, color_jpeg = cv2.imencode(".jpg", color)
                        color_bytes = color_jpeg.tobytes()
                        # 深度 float32，单位米
                        depth = np.asarray(depth, dtype=np.float32)
                        h, w = depth.shape
                        depth_bytes = depth.tobytes()
                        # 相机内参 K: 3x3 float64，按行优先打平
                        K_flat = np.asarray(K, dtype=np.float64).reshape(-1)
                        K_bytes = K_flat.tobytes()
                        # 发送: OK\n + 9*8B K + 4B len + color + 4B H + 4B W + depth
                        conn.sendall(b"OK\n")
                        conn.sendall(K_bytes)
                        conn.sendall(struct.pack("<I", len(color_bytes)))
                        conn.sendall(color_bytes)
                        conn.sendall(struct.pack("<II", h, w))
                        conn.sendall(depth_bytes)
                        print(f"[CAPTURE] 已发送 K, color={len(color_bytes)}B depth={h}x{w} float32")

                    elif cmd == "POSE" and len(parts) == 8:
                        try:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            rx, ry, rz = float(parts[4]), float(parts[5]), float(parts[6])
                            grip = float(parts[7])
                            if robot is None:
                                conn.sendall(b"ERR robot not available\n")
                            else:
                                # 要求: x,y,z,rx,ry,rz 为相机坐标系下的位置与姿态
                                if R_cam2base is None or t_cam2base is None:
                                    conn.sendall(b"ERR no hand-eye calibration on lower_server\n")
                                    continue

                                # 位置 cam -> base
                                p_cam_m = np.array([x, y, z], dtype=np.float64) / 1000.0
                                p_base_m = R_cam2base @ p_cam_m + t_cam2base
                                bx, by, bz = (p_base_m * 1000.0).tolist()

                                # 姿态 cam -> base
                                rx_c, ry_c, rz_c = np.deg2rad([rx, ry, rz])
                                R_ee_cam = _euler_zyx_to_matrix(rx_c, ry_c, rz_c)
                                R_ee_base = R_cam2base @ R_ee_cam
                                rx_b, ry_b, rz_b = _matrix_to_euler_zyx(R_ee_base)
                                rx_deg, ry_deg, rz_deg = np.rad2deg([rx_b, ry_b, rz_b])

                                robot.send_pose_gripper_cmd(bx, by, bz, rx_deg, ry_deg, rz_deg, grip, vec=50)
                                conn.sendall(b"OK\n")
                                print(
                                    f"[POSE] cam_xyz_mm=({x},{y},{z}) -> base_xyz_mm=({bx:.3f},{by:.3f},{bz:.3f}) "
                                    f"cam_rpy_deg=({rx:.3f},{ry:.3f},{rz:.3f}) -> "
                                    f"base_rpy_deg=({rx_deg:.3f},{ry_deg:.3f},{rz_deg:.3f}) grip={grip}"
                                )
                        except (ValueError, IndexError) as e:
                            conn.sendall(f"ERR {e}\n".encode("utf-8"))
                    else:
                        conn.sendall(b"ERR unknown or invalid command\n")
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            print(f"[TCP] 连接异常: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="下位机 TCP 服务：截图 + 机械臂位姿")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8888, help="监听端口")
    parser.add_argument("--color-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--info-topic", default="/camera/color/camera_info")
    parser.add_argument("--no-robot", action="store_true", help="不连接机械臂，仅提供截图")
    parser.add_argument("--R-path", type=str, default="/home/common/eye_hand/R.txt",
                        help="手眼标定旋转矩阵 txt，shape=(3,3)，表示 p_base = R @ p_cam + t")
    parser.add_argument("--t-path", type=str, default="/home/common/eye_hand/t.txt",
                        help="手眼标定平移向量 txt，shape=(3,) 或 (3,1)")
    args = parser.parse_args()

    rclpy.init()
    rgbd_node = RGBDSyncNode(args.color_topic, args.depth_topic, args.info_topic)

    # 加载手眼标定，用于在下位机完成相机坐标 -> base 坐标变换
    R_cam2base = None
    t_cam2base = None
    if args.R_path and args.t_path:
        try:
            R_cam2base = _load_txt_matrix(args.R_path, (3, 3))
            t_cam2base = _load_txt_matrix(args.t_path, (3,)).reshape(3)
            print("[手眼标定] 已加载 R, t ，将在下位机完成 p_base = R @ p_cam + t 变换")
            print("R =\n", R_cam2base)
            print("t =", t_cam2base)
        except Exception as e:
            print(f"[手眼标定] 加载失败: {e}")
    robot = None
    if not args.no_robot and PiperX_Arm is not None:
        robot = PiperX_Arm()
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(robot)
        robot_spin = threading.Thread(target=executor.spin, daemon=True)
        robot_spin.start()
        print("[机械臂] PiperX_Arm 已连接")
    else:
        if args.no_robot:
            print("[机械臂] 已禁用 (--no-robot)")
        else:
            print("[机械臂] PiperX_Arm 未找到，仅提供截图")

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(rgbd_node)
    ros_spin = threading.Thread(target=executor.spin, daemon=True)
    ros_spin.start()

    # 等待至少一帧彩图+深度(+内参)
    import time
    for _ in range(100):
        c, d, K = rgbd_node.get_latest()
        if c is not None and d is not None:
            if K is not None:
                print("[相机] 已收到首帧彩图、深度图和 CameraInfo")
            else:
                print("[相机] 已收到首帧彩图与深度图（CameraInfo 仍为空）")
            break
        time.sleep(0.05)
    else:
        print("[相机] 等待 /camera/color/image_raw 与 /camera/aligned_depth_to_color/image_raw ...")

    run_tcp_server(args.host, args.port, rgbd_node, robot,
                   R_cam2base=R_cam2base, t_cam2base=t_cam2base)


if __name__ == "__main__":
    main()
