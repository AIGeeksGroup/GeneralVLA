#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os

# slove X11 monitor make kernel died
if "SESSION_MANAGER" in os.environ:
    del os.environ["SESSION_MANAGER"]
    
sys.path.append("/opt/ros/jazzy/lib/python3.12/site-packages")
sys.path.append("/py_venvs/brainarm/lib/python3.12/site-packages")
sys.path.append(os.environ.get("HOME")+"/brainarm-ws/install/piper_msgs/lib/python3.12/site-packages")

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ["QT_FONT_DPI"] = "96"

import argparse
import time
import numpy as np
import cv2
import rclpy
import threading
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters

# -------------------- 数学工具 --------------------
def quat_to_rotmat(x, y, z, w):
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    R = np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)]
    ], dtype=np.float64)
    return R

def solve_rigid_transform_kabsch(points_cam, points_base):
    A = np.asarray(points_cam, dtype=np.float64)
    B = np.asarray(points_base, dtype=np.float64)
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t

def compute_rmse(points_cam, points_base, R, t):
    A = np.asarray(points_cam, dtype=np.float64)
    B = np.asarray(points_base, dtype=np.float64)
    B_pred = (R @ A.T).T + t
    err = np.linalg.norm(B_pred - B, axis=1)
    rmse = np.sqrt(np.mean(err ** 2))
    return rmse, err

# -------------------- Piper 读取 --------------------
class PiperPoseReader:
    def __init__(self, pose_topic="end_pose_stamped", unit_mm=True):
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("piper_eyehand_pose_reader")
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
                pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64)
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

# -------------------- ROS 订阅相机 --------------------
class L515ClickPointReader:
    def __init__(self, depth_radius=4, min_depth=0.25, max_depth=3.0,
                 color_topic="/camera/color/image_raw",
                 depth_topic="/camera/aligned_depth_to_color/image_raw",
                 info_topic="/camera/color/camera_info",
                 window_name="L515 Color",
                 depth_window_name="L515 Depth"):


        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("l515_click_reader")
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
        color = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        if depth_msg.encoding == '16UC1':
            depth = depth.astype(np.float32) / 1000.0  # mm -> m
        else:
            depth = depth.astype(np.float32)

        with self.lock:
            self.color_img = color
            self.depth_img = depth

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self.lock:
                if self.color_img is None or self.depth_img is None or self.fx is None:
                    self.click_status_text = "Camera data not ready"
                    return
                depth = self.depth_img.copy()
                fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

            depth_val = self.get_valid_depth_median(depth, x, y, self.depth_radius, self.min_depth, self.max_depth)
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
            print(f"[已锁定相机点] Pixel=({x},{y}) Cam XYZ={np.round(self.clicked_cam_point,4)}")

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
        txt1 = f"Pixel(u,v)=({u},{v})  Locked Depth={clicked_depth_val:.4f}m"
        txt2 = f"Locked Cam XYZ(m): [{clicked_cam_point[0]:.4f}, {clicked_cam_point[1]:.4f}, {clicked_cam_point[2]:.4f}]"
        cv2.putText(vis, txt0, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(vis, txt1, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(vis, txt2, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        return clicked_cam_point, vis, depth_vis

# -------------------- 主程序 --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--depth-radius", type=int, default=4)
    parser.add_argument("--min-depth", type=float, default=0.25)
    parser.add_argument("--max-depth", type=float, default=3.0)

    parser.add_argument("--color-topic", type=str, default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", type=str, default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--info-topic", type=str, default="/camera/color/camera_info")

    parser.add_argument("--tool-point-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--output", type=str, default="eye_to_hand_result.npz")
    parser.add_argument("--min-step-cam", type=float, default=0.005)
    parser.add_argument("--min-step-base", type=float, default=0.005)
    parser.add_argument("--print-cam-interval", type=float, default=0.2)
    args = parser.parse_args()

    tool_offset = np.array(args.tool_point_offset, dtype=np.float64).reshape(3)

    tf_reader = PiperPoseReader(pose_topic="end_pose_stamped", unit_mm=True)
    cam_reader = L515ClickPointReader(
        depth_radius=args.depth_radius,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        color_topic=args.color_topic,
        depth_topic=args.depth_topic,
        info_topic=args.info_topic
    )

    cam_points = []
    base_points = []
    last_print_t = 0.0

    try:
        cam_reader.start()
        print("\n=== 开始采样 ===")
        print("鼠标点击标定点时锁定当前相机三维坐标；机械臂末端移动到被测点后按 's' 保存一组样本；按 'c' 清除；按 'q' 退出。\n")

        while True:
            p_cam, vis, depth_vis = cam_reader.get_click_point_3d()

            if p_cam is not None:
                now = time.time()
                if args.print_cam_interval <= 0 or (now - last_print_t) >= args.print_cam_interval:
                    print(f"[锁定相机坐标] X={p_cam[0]:.4f} m, Y={p_cam[1]:.4f} m, Z={p_cam[2]:.4f} m")
                    last_print_t = now

            p_base_preview = None
            if p_cam is not None:
                try:
                    p_base_preview = tf_reader.get_calib_point_in_base(tool_offset)
                except Exception as e:
                    if vis is not None:
                        cv2.putText(vis, f"TF lookup failed: {str(e)[:50]}", (20, 135),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            if vis is not None:
                cv2.putText(vis, f"Samples: {len(cam_points)}/{args.samples}", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
                if p_base_preview is not None:
                    txt2 = f"Current Base XYZ(m): [{p_base_preview[0]:.4f}, {p_base_preview[1]:.4f}, {p_base_preview[2]:.4f}]"
                    cv2.putText(vis, txt2, (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 200, 0), 2)
                cv2.imshow(cam_reader.window_name, vis)

            if depth_vis is not None:
                cv2.imshow(cam_reader.depth_window_name, depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("退出。")
                break
            if key == ord('c'):
                cam_reader.reset_click()
                print("已清除点击点。")
            if key == ord('s'):
                if p_cam is None:
                    print("[跳过] 尚未点击并锁定有效的相机三维点。")
                    continue
                try:
                    p_base = tf_reader.get_calib_point_in_base(tool_offset)
                except Exception as e:
                    print(f"[跳过] 读取机械臂末端位姿失败: {e}")
                    continue
                if len(cam_points) > 0:
                    dc = np.linalg.norm(p_cam - cam_points[-1])
                    db = np.linalg.norm(p_base - base_points[-1])
                    if dc < args.min_step_cam and db < args.min_step_base:
                        print(f"[跳过] 位姿变化太小: d_cam={dc:.4f}, d_base={db:.4f}")
                        continue
                cam_points.append(p_cam.copy())
                base_points.append(p_base.copy())
                print(f"[已采样 {len(cam_points)}/{args.samples}] locked_cam={np.round(p_cam,4)} current_base={np.round(p_base,4)}")
                cam_reader.reset_click()

                if len(cam_points) >= args.samples:
                    print("采样完成。")
                    break

        if len(cam_points) < 3:
            print("有效样本不足 3 组，无法标定。")
            return

        R, t = solve_rigid_transform_kabsch(cam_points, base_points)
        rmse, errs = compute_rmse(cam_points, base_points, R, t)

        print("\n========== 标定结果 ==========")
        print("p_base = R @ p_cam + t")
        print("R:\n", R)
        print("t:", t)
        print(f"RMSE: {rmse:.6f}")

        np.savez(args.output, R=R, t=t,
                 cam_points=np.array(cam_points),
                 base_points=np.array(base_points),
                 rmse=rmse, errors=errs)
        print(f"\n结果已保存到: {args.output}")

    finally:
        cam_reader.stop()
        tf_reader.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()