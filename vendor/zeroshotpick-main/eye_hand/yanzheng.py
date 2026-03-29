#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("/opt/ros/jazzy/lib/python3.12/site-packages")

import os
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ["QT_FONT_DPI"] = "96"

import argparse
import time
import threading
import numpy as np
import cv2
import rclpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters


# ==================== 数学工具 ====================
def quat_to_rotmat(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    R = np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)]
    ], dtype=np.float64)
    return R


def load_txt_matrix(path, shape):
    arr = np.loadtxt(path, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64).reshape(shape)
    return arr


def transform_cam_to_base(p_cam, R, t):
    return R @ p_cam + t


# ==================== Piper 读取 ====================
class PiperPoseReader:
    def __init__(self, pose_topic="end_pose_stamped", unit_mm=True):
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("piper_eyehand_verify_pose_reader")
        self.pose_topic = pose_topic
        self.unit_mm = unit_mm
        self.latest_pose = None
        self.lock = threading.Lock()

        self.node.create_subscription(PoseStamped, self.pose_topic, self._pose_cb, 10)

        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

    def _pose_cb(self, msg):
        with self.lock:
            self.latest_pose = msg

    def get_tool_pose(self, timeout_sec=0.2):
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            with self.lock:
                msg = self.latest_pose
            if msg is not None:
                pos = np.array(
                    [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                    dtype=np.float64
                )
                if self.unit_mm:
                    pos = pos / 1000.0
                q = msg.pose.orientation
                Rm = quat_to_rotmat(q.x, q.y, q.z, q.w)
                return pos, Rm
            time.sleep(0.005)
        raise TimeoutError("No pose received from end_pose_stamped")

    def get_calib_point_in_base(self, tool_point_offset):
        t_base_tool, R_base_tool = self.get_tool_pose()
        return t_base_tool + R_base_tool @ tool_point_offset.reshape(3)

    def close(self):
        self.executor.shutdown()
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ==================== ROS 订阅相机 ====================
class L515ClickPointReader:
    def __init__(self,
                 depth_radius=4,
                 min_depth=0.25,
                 max_depth=3.0,
                 color_topic="/camera/color/image_raw",
                 depth_topic="/camera/aligned_depth_to_color/image_raw",
                 info_topic="/camera/color/camera_info",
                 window_name="L515 Color Verify",
                 depth_window_name="L515 Depth Verify"):

        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node("l515_click_verify_reader")
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
        ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=10,
            slop=0.05
        )
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

        depth_val = self.get_valid_depth_median(
            depth, x, y,
            radius=self.depth_radius,
            min_d=self.min_depth,
            max_d=self.max_depth
        )

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
                ready = (
                    self.color_img is not None and
                    self.depth_img is not None and
                    self.fx is not None
                )
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
    parser.add_argument("--depth-radius", type=int, default=4)
    parser.add_argument("--min-depth", type=float, default=0.25)
    parser.add_argument("--max-depth", type=float, default=3.0)

    parser.add_argument("--color-topic", type=str, default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", type=str, default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--info-topic", type=str, default="/camera/color/camera_info")
    parser.add_argument("--pose-topic", type=str, default="end_pose_stamped")

    parser.add_argument("--tool-point-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])

    parser.add_argument("--R-path", type=str, default="R.txt")
    parser.add_argument("--t-path", type=str, default="t.txt")
    parser.add_argument("--unit-mm", action="store_true", default=True,
                        help="机械臂 PoseStamped 位置是否为 mm，默认 True")
    args = parser.parse_args()

    tool_offset = np.array(args.tool_point_offset, dtype=np.float64).reshape(3)
    R = load_txt_matrix(args.R_path, (3, 3))
    t = load_txt_matrix(args.t_path, (3,)).reshape(3)

    print("\n========== 加载验证矩阵 ==========")
    print("使用变换: p_base = R @ p_cam + t")
    print("R =\n", R)
    print("t =", t)

    tf_reader = PiperPoseReader(pose_topic=args.pose_topic, unit_mm=args.unit_mm)
    cam_reader = L515ClickPointReader(
        depth_radius=args.depth_radius,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        color_topic=args.color_topic,
        depth_topic=args.depth_topic,
        info_topic=args.info_topic
    )

    last_pred_base = None
    last_real_base = None
    last_err = None

    try:
        cam_reader.start()
        print("\n=== 开始验证 ===")
        print("操作说明：")
        print("1) 鼠标左键点击彩色图中的标定点，锁定相机坐标")
        print("2) 机械臂移动到对应真实点后，按 's' 读取真实机械臂基座坐标并比对")
        print("3) 按 'c' 清除当前点击点")
        print("4) 按 'q' 退出\n")

        while True:
            p_cam, vis, depth_vis = cam_reader.get_click_point_3d()

            p_base_pred = None
            if p_cam is not None:
                p_base_pred = transform_cam_to_base(p_cam, R, t)
                last_pred_base = p_base_pred.copy()

            if vis is not None:
                if p_base_pred is not None:
                    txt3 = f"Pred Base XYZ(m): [{p_base_pred[0]:.4f}, {p_base_pred[1]:.4f}, {p_base_pred[2]:.4f}]"
                    cv2.putText(vis, txt3, (20, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                if last_real_base is not None:
                    txt4 = f"Real Base XYZ(m): [{last_real_base[0]:.4f}, {last_real_base[1]:.4f}, {last_real_base[2]:.4f}]"
                    cv2.putText(vis, txt4, (20, 175),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if last_err is not None:
                    err_norm_mm = np.linalg.norm(last_err) * 1000.0
                    txt5 = f"Err XYZ(m): [{last_err[0]:.4f}, {last_err[1]:.4f}, {last_err[2]:.4f}]"
                    txt6 = f"Err Norm: {err_norm_mm:.2f} mm"
                    cv2.putText(vis, txt5, (20, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(vis, txt6, (20, 245),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow(cam_reader.window_name, vis)

            if depth_vis is not None:
                cv2.imshow(cam_reader.depth_window_name, depth_vis)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("退出。")
                break

            elif key == ord("c"):
                cam_reader.reset_click()
                last_pred_base = None
                last_real_base = None
                last_err = None
                print("已清除当前点击点和比对结果。")

            elif key == ord("s"):
                if p_cam is None:
                    print("[跳过] 尚未点击并锁定有效的相机三维点。")
                    continue

                try:
                    p_base_real = tf_reader.get_calib_point_in_base(tool_offset)
                except Exception as e:
                    print(f"[跳过] 读取机械臂末端位姿失败: {e}")
                    continue

                p_base_pred = transform_cam_to_base(p_cam, R, t)
                err = p_base_pred - p_base_real

                last_pred_base = p_base_pred.copy()
                last_real_base = p_base_real.copy()
                last_err = err.copy()

                print("\n================ 验证结果 ================")
                print(f"Pixel(u,v): {cam_reader.clicked_uv}")
                print(f"Cam XYZ(m):      [{p_cam[0]:.6f}, {p_cam[1]:.6f}, {p_cam[2]:.6f}]")
                print(f"Pred Base XYZ(m):[{p_base_pred[0]:.6f}, {p_base_pred[1]:.6f}, {p_base_pred[2]:.6f}]")
                print(f"Real Base XYZ(m):[{p_base_real[0]:.6f}, {p_base_real[1]:.6f}, {p_base_real[2]:.6f}]")
                print(f"Err XYZ(m):      [{err[0]:.6f}, {err[1]:.6f}, {err[2]:.6f}]")
                print(f"Err Norm:        {np.linalg.norm(err):.6f} m  ({np.linalg.norm(err)*1000.0:.2f} mm)")
                print("==========================================\n")

    finally:
        cam_reader.stop()
        tf_reader.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
