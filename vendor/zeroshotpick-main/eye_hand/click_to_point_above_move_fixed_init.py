#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("/opt/ros/jazzy/lib/python3.12/site-packages")

import os
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ["QT_FONT_DPI"] = "96"

# slove X11 monitor make kernel died
if "SESSION_MANAGER" in os.environ:
    del os.environ["SESSION_MANAGER"]

import time
import threading
import argparse
import numpy as np
import cv2
import rclpy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
from piperx_arm_template import PiperX_Arm


# ==================== 数学工具 ====================
def load_txt_matrix(path, shape):
    arr = np.loadtxt(path, dtype=np.float64)
    return np.asarray(arr, dtype=np.float64).reshape(shape)


def transform_cam_to_base(p_cam, R, t):
    return R @ p_cam + t


# ==================== 相机点击读取 ====================
class L515ClickPointReader:
    def __init__(self,
                 depth_radius=4,
                 min_depth=0.25,
                 max_depth=3.0,
                 color_topic="/camera/color/image_raw",
                 depth_topic="/camera/aligned_depth_to_color/image_raw",
                 info_topic="/camera/color/camera_info",
                 window_name="L515 Click Move",
                 depth_window_name="L515 Depth"):
        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node("l515_click_move_reader")
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.color_img = None
        self.depth_img = None
        self.fx = self.fy = self.cx = self.cy = None

        self.depth_radius = depth_radius
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.window_name = window_name
        self.depth_window_name = depth_window_name

        self.clicked_uv = None
        self.clicked_cam_point = None
        self.clicked_depth_val = None
        self.click_status_text = "Click on target point"

        color_sub = message_filters.Subscriber(self.node, Image, color_topic)
        depth_sub = message_filters.Subscriber(self.node, Image, depth_topic)
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=10, slop=0.05)
        ts.registerCallback(self._image_cb)

        self.node.create_subscription(CameraInfo, info_topic, self._info_cb, 10)

        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

        cv2.namedWindow(self.window_name)
        cv2.namedWindow(self.depth_window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def _info_cb(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def _image_cb(self, color_msg, depth_msg):
        color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if depth_msg.encoding == "16UC1":
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)

        with self.lock:
            self.color_img = color
            self.depth_img = depth

    @staticmethod
    def get_valid_depth_median(depth_img, u, v, radius=4, min_d=0.25, max_d=3.0):
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

        with self.lock:
            if self.color_img is None or self.depth_img is None or self.fx is None:
                self.click_status_text = "Camera data not ready"
                return
            depth = self.depth_img.copy()
            fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        depth_val = self.get_valid_depth_median(depth, x, y,
                                                radius=self.depth_radius,
                                                min_d=self.min_depth,
                                                max_d=self.max_depth)

        self.clicked_uv = (x, y)
        if depth_val <= 0:
            self.clicked_cam_point = None
            self.clicked_depth_val = None
            self.click_status_text = "Depth invalid"
            return

        X = (x - cx) * depth_val / fx
        Y = (y - cy) * depth_val / fy
        Z = depth_val
        self.clicked_cam_point = np.array([X, Y, Z], dtype=np.float64)
        self.clicked_depth_val = depth_val
        self.click_status_text = "Point locked"
        print(f"[已锁定相机点] Pixel=({x},{y}) Cam XYZ={np.round(self.clicked_cam_point, 6)}")

    def reset_click(self):
        self.clicked_uv = None
        self.clicked_cam_point = None
        self.clicked_depth_val = None
        self.click_status_text = "Click on target point"

    def start(self):
        t0 = time.time()
        while rclpy.ok():
            with self.lock:
                ready = self.color_img is not None and self.depth_img is not None and self.fx is not None
            if ready:
                break
            if time.time() - t0 > 5:
                print("等待相机话题...")
                t0 = time.time()
            time.sleep(0.05)

    def stop(self):
        self.executor.shutdown()
        self.node.destroy_node()

    def _make_depth_vis(self, depth):
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_norm = (depth - self.min_depth) / (self.max_depth - self.min_depth)
        depth_norm = np.clip(depth_norm, 0, 1)
        depth_vis = (depth_norm * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        return depth_vis

    def get_click_point_3d(self):
        with self.lock:
            if self.color_img is None or self.depth_img is None or self.fx is None:
                return None, None, None
            color = self.color_img.copy()
            depth = self.depth_img.copy()
            clicked_uv = self.clicked_uv
            clicked_cam_point = None if self.clicked_cam_point is None else self.clicked_cam_point.copy()
            clicked_depth_val = self.clicked_depth_val
            status_text = self.click_status_text

        vis = color.copy()
        depth_vis = self._make_depth_vis(depth)

        if clicked_uv is None:
            cv2.putText(vis, status_text, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return None, vis, depth_vis

        u, v = clicked_uv
        cv2.circle(vis, (u, v), 6, (0, 255, 0), -1)
        cv2.circle(depth_vis, (u, v), 6, (0, 255, 0), -1)

        if clicked_cam_point is None or clicked_depth_val is None:
            cv2.putText(vis, status_text, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return None, vis, depth_vis

        txt0 = f"Status: {status_text}"
        txt1 = f"Pixel(u,v)=({u},{v}) Locked Depth={clicked_depth_val:.4f}m"
        txt2 = f"Locked Cam XYZ(m): [{clicked_cam_point[0]:.4f}, {clicked_cam_point[1]:.4f}, {clicked_cam_point[2]:.4f}]"
        cv2.putText(vis, txt0, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, txt1, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, txt2, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return clicked_cam_point, vis, depth_vis


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--R-path", type=str, default="R.txt")
    parser.add_argument("--t-path", type=str, default="t.txt")
    parser.add_argument("--hover-mm", type=float, default=5.0)
    parser.add_argument("--move-vec", type=int, default=30)
    parser.add_argument("--return-vec", type=int, default=30)
    parser.add_argument("--depth-radius", type=int, default=4)
    parser.add_argument("--min-depth", type=float, default=0.25)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--color-topic", type=str, default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", type=str, default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--info-topic", type=str, default="/camera/color/camera_info")
    parser.add_argument("--settle-sec", type=float, default=2.0)
    args = parser.parse_args()

    R = load_txt_matrix(args.R_path, (3, 3))
    t = load_txt_matrix(args.t_path, (3,)).reshape(3) 

    print("\n========== 已加载手眼标定 ==========")
    print("使用变换: p_base = R @ p_cam + t")
    print("R =\n", R)
    print("t =", t)

    rclpy.init(args=None)
    cam_reader = L515ClickPointReader(
        depth_radius=args.depth_radius,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        color_topic=args.color_topic,
        depth_topic=args.depth_topic,
        info_topic=args.info_topic,
    )
    robot = PiperX_Arm()

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(robot)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    def send_pose_cmd_reliably(x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg, grip, vec, repeat=8, dt=0.05):
        """重复发布几次，避免单次按键消息被漏收。"""
        for i in range(repeat):
            robot.send_pose_gripper_cmd(x_mm, y_mm, z_mm,  
                                        rx_deg, ry_deg, rz_deg, grip,
                                        vec=vec)
            print(f"[publish {i+1}/{repeat}] XYZ(mm)=({x_mm:.2f}, {y_mm:.2f}, {z_mm:.2f}) RPY(deg)=({rx_deg:.2f}, {ry_deg:.2f}, {rz_deg:.2f}) grip={grip:.2f} vec={vec}")
            time.sleep(dt)

    init_pose = [138.26, -2.74, 160.77, -171.41, -1.31, -82.54, -0.00]  # [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg, grip]
    last_target_base = None

    try:
        cam_reader.start()
        print("\n=== 操作说明 ===")  
        print("初始位姿已写死固定，不再从机械臂当前状态读取或修改")
        print("空格键: 回到写死的初始位姿")
        print("鼠标左键: 点击图像中的目标点，锁定相机三维点")
        print("回车键: 机械臂末端移动到该点在基座系下方位姿的上方 5mm")
        print("c: 清除当前点击点")
        print("q: 退出\n")

        while True:
            p_cam, vis, depth_vis = cam_reader.get_click_point_3d()

            p_base = None
            if p_cam is not None:
                p_base = transform_cam_to_base(p_cam, R, t)
                last_target_base = p_base.copy()

            if vis is not None:
                if init_pose is None:
                    init_txt = "Init Pose: NOT SET"
                else:
                    init_txt = (
                        f"Init(mm,deg): [{init_pose[0]:.1f}, {init_pose[1]:.1f}, {init_pose[2]:.1f}] "
                        f"[{init_pose[3]:.1f}, {init_pose[4]:.1f}, {init_pose[5]:.1f}] grip={init_pose[6]:.1f}"
                    )
                cv2.putText(vis, init_txt, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

                if p_base is not None:
                    txt3 = f"Pred Base XYZ(m): [{p_base[0]:.4f}, {p_base[1]:.4f}, {p_base[2]:.4f}]"
                    txt4 = f"Move Target(mm): [{p_base[0]*1000:.1f}, {p_base[1]*1000:.1f}, {(p_base[2]*1000 + args.hover_mm):.1f}]"
                    cv2.putText(vis, txt3, (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
                    cv2.putText(vis, txt4, (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

                cv2.imshow(cam_reader.window_name, vis)

            if depth_vis is not None:
                cv2.imshow(cam_reader.depth_window_name, depth_vis)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("退出。")
                break

            elif key == ord('c'):
                cam_reader.reset_click()
                last_target_base = None
                print("已清除当前点击点。")

            elif key == 32:  # 空格
                print("[回到写死初始位姿]")
                print(f"XYZ(mm)=({init_pose[0]:.2f}, {init_pose[1]:.2f}, {init_pose[2]:.2f})")
                print(f"RPY(deg)=({init_pose[3]:.2f}, {init_pose[4]:.2f}, {init_pose[5]:.2f}) grip={init_pose[6]:.2f}")
                send_pose_cmd_reliably(init_pose[0], init_pose[1], init_pose[2],
                                       init_pose[3], init_pose[4], init_pose[5], init_pose[6],
                                       vec=args.return_vec)
                time.sleep(args.settle_sec)

            elif key == 13:  # 回车
                if p_cam is None or last_target_base is None:
                    print("[跳过] 请先在图像中点击一个有效点。")
                    continue
                target_x_mm = last_target_base[0] * 1000.0
                target_y_mm = last_target_base[1] * 1000.0
                target_z_mm = last_target_base[2] * 1000.0 + args.hover_mm

                print("[执行移动]")
                print(f"Cam XYZ(m): {np.round(p_cam, 6)}")
                print(f"Base XYZ(m): {np.round(last_target_base, 6)}")
                print(f"Move XYZ(mm): ({target_x_mm:.2f}, {target_y_mm:.2f}, {target_z_mm:.2f})")
                print(f"Fixed RPY(deg): ({init_pose[3]:.2f}, {init_pose[4]:.2f}, {init_pose[5]:.2f})")

                send_pose_cmd_reliably(target_x_mm, target_y_mm, target_z_mm,
                                        init_pose[3], init_pose[4], init_pose[5], init_pose[6],
                                        vec=args.move_vec)
                time.sleep(args.settle_sec)

    finally:
        cam_reader.stop()
        executor.shutdown()
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
