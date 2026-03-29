#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的 zeroshot 应用入口：

  - 虚拟数据最小回环测试：复用 test_zeroshot.py 的离线 RGBD 流程
  - 实机 RGBD 抓取规划：通过 UpperClient 从下位机获取一帧 RGBD (+K)，走 ZeroShotPlanner

用法示例：

  # 1) 虚拟数据最小回环测试（与 test_zeroshot 等价）
  python3 zeroshot_apps.py --mode virtual \\
      --rgb rgb.jpg --depth depth.png

  # 2) 实机在线 RGBD 抓取规划（仅计算位姿，不发给机械臂）
  python3 zeroshot_apps.py --mode online \\
      --host 10.5.23.176 --port 8888 \\
      --pick_text "抓起桌面上最显眼的一个物体" \\
      --place_text "把它放到桌面右下角的空白区域"
"""

import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np

from zeroshot_pipeline import (
    ZeroShotPlanner,
    VLMClient,
    SAMSegmenter,
    GraspEstimator,
    backproject_pixel_to_3d,
    MotionStep,
    se3_from_grasp_in_cam,
)
from upper_client import UpperClient
from test_zeroshot import (
    OUTPUT_DIR,
    visualize_vlm_bboxes,
    visualize_sam_mask,
    visualize_cloud_and_grasp_o3d,
)


def _rotation_matrix_to_euler_zyx_deg(R: np.ndarray) -> Tuple[float, float, float]:
    """
    将 3x3 旋转矩阵转成 ZYX 顺序的欧拉角 (rx, ry, rz)，单位：度。
    假设:
      R = Rz(rz) @ Ry(ry) @ Rx(rx)
    对应下位机 {rx, ry, rz} = (roll, pitch, yaw)。
    """
    assert R.shape == (3, 3)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        rz = np.arctan2(R[1, 0], R[0, 0])
        ry = np.arctan2(-R[2, 0], sy)
        rx = np.arctan2(R[2, 1], R[2, 2])
    else:
        # 退化情况：pitch 接近 +/-90deg
        rz = np.arctan2(-R[0, 1], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rx = 0.0
    rx_deg, ry_deg, rz_deg = np.degrees([rx, ry, rz])
    return float(rx_deg), float(ry_deg), float(rz_deg)


def _euler_zyx_deg_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """
    将 (rx, ry, rz) roll-pitch-yaw 角（度）转换为旋转矩阵，约定:
      R = Rz(rz) @ Ry(ry) @ Rx(rx)
    """
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cr, sr = np.cos(rx), np.sin(rx)
    cp, sp = np.cos(ry), np.sin(ry)
    cy, sy = np.cos(rz), np.sin(rz)
    Rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    Ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float64,
    )
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float64,
    )
    return (Rz @ Ry @ Rx).astype(np.float32)


def execute_motion_sequence(
    client: UpperClient, steps: List[MotionStep], initial_grip: float
) -> float:
    """
    顺序执行一系列 MotionStep，统一处理：
      - 是否在执行前向用户确认（prompt）
      - “保持当前夹爪开合状态”/显式设置夹爪开合量
      - 某些 step 需要依赖上一个 step 已成功执行（require_prev_executed）

    返回最终的夹爪开合量（方便后续作为初始状态继续扩展序列）。
    """
    current_grip = initial_grip
    last_executed = False

    for step in steps:
        if step.require_prev_executed and not last_executed:
            print(
                f"[online] 上一步未执行或失败，自动跳过依赖步骤：{step.desc}"
            )
            last_executed = False
            continue

        if step.prompt:
            try:
                user_in = input(step.prompt).strip().lower()
            except EOFError:
                user_in = "n"
            if user_in != "":
                print(f"[online] 本次已选择跳过步骤：{step.desc}")
                last_executed = False
                continue

        grip_to_send = current_grip if step.keep_grip else step.grip
        if not step.keep_grip:
            current_grip = step.grip

        ok = _send_motion_step_in_cam(
            client,
            T_cam=step.T_cam,
            grip=grip_to_send,
            desc=step.desc,
            apply_graspnet_corr=step.apply_graspnet_corr,
        )
        if not ok:
            print("[online] 执行步骤失败，中止后续序列。")
            last_executed = False
            break

        last_executed = True

    return current_grip


def _send_motion_step_in_cam(
    client: UpperClient,
    T_cam: np.ndarray,
    grip: float,
    desc: str,
    apply_graspnet_corr: bool = True,
) -> bool:
    """
    执行单个“相机系下的移动 + 夹爪”步骤：
      - 输入：相机系齐次矩阵 T_cam（4x4）、夹爪开合量 grip
      - 内部：做与 GraspNet 一致的姿态矫正 + mm/deg 转换后，通过 UpperClient 发送
    """
    t_cam = T_cam[:3, 3].astype(float)
    R_cam_raw = T_cam[:3, :3].astype(float)
    if apply_graspnet_corr:
        R_for_send = (_R_GRASPNET_TO_CAM @ R_cam_raw).astype(float)
    else:
        # 对于已经在「相机末端坐标系」下表达的姿态（例如人工标定得到的初始位姿），
        # 不再叠加 GraspNet -> Cam 的校正矩阵，直接使用用户给定旋转。
        R_for_send = R_cam_raw
    rx_deg, ry_deg, rz_deg = _rotation_matrix_to_euler_zyx_deg(R_for_send)
    x_mm, y_mm, z_mm = (t_cam * 1000.0).tolist()

    # 与在线调试阶段保持一致，仅保留 yaw，自由设定 roll/pitch 为 0
    rx_deg = 0.0
    ry_deg = 0.0

    print(
        f"\n[online] 发送相机系“{desc}”到下位机 (单位: mm, deg):\n"
        f"  XYZ = ({x_mm:.2f}, {y_mm:.2f}, {z_mm:.2f})\n"
        f"  RPY = ({rx_deg:.2f}, {ry_deg:.2f}, {rz_deg:.2f})  grip={grip:.2f}"
    )

    try:
        client.send_pose(x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg, grip)
    except Exception as e:
        print(f"[online] 发送“{desc}”到下位机失败: {e}")
        return False

    print(f"[online] 已完成{desc}。")
    return True


# -------------------------
# GraspNet 坐标系 -> 相机坐标系 姿态矫正（规范成“绕 x 轴 90°”）
# -------------------------
#
# 通过你给出的两组 cam_rpy，可以看出 GraspNet 的抓取局部坐标系相对
# 我们期望的相机末端坐标系主要是绕自身 x 轴的一次大角度旋转。
# 为了避免依赖不精确的数值，我们把这个偏置规范化为：
#   R_graspnet_to_cam = Rx(+90°)
# 即假定 GraspNet 的 x 轴与我们期望的相机 EE x 轴重合，
# 但其 y/z 轴与 EE 坐标系相差一发 90° 的滚转。
#
# 实际发送时，对每一帧 GraspNet 给出的 R_cam_grasp 使用：
#   R_cam_corrected = R_graspnet_to_cam @ R_cam_grasp
# 再拆欧拉发给下位机，这样既保留 GraspNet 的姿态变化，又用一个规范的 90° 轴变换
# 把其内部抓取坐标系对齐到我们使用的机械臂末端坐标系。
_R_GRASPNET_TO_CAM = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def _ensure_file_exist(path: str, desc: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(f"{desc} 文件不存在: {path}")


def _load_virtual_rgbd(rgb_path: str, depth_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """从磁盘加载虚拟 RGBD（与 test_zeroshot 中逻辑保持一致的简化版）"""
    _ensure_file_exist(rgb_path, "RGB")
    _ensure_file_exist(depth_path, "深度")

    color_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise SystemExit(f"读取 RGB 失败: {rgb_path}")

    raw_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if raw_depth is None:
        raise SystemExit(f"读取深度图失败: {depth_path}")

    if raw_depth.dtype != np.uint16:
        print(f"[警告] 深度图并非 16-bit, 当前 dtype={raw_depth.dtype}")
    depth_m = raw_depth.astype(np.float32) / 1000.0  # mm -> m

    h_d, w_d = depth_m.shape
    h_c, w_c, _ = color_bgr.shape
    if (h_d, w_d) != (h_c, w_c):
        print(
            f"[INFO] RGB 分辨率为 {h_c}x{w_c}, 深度分辨率为 {h_d}x{w_d}, "
            "对深度图进行插值以对齐到 RGB 分辨率。"
        )
        depth_m = cv2.resize(
            depth_m,
            (w_c, h_c),
            interpolation=cv2.INTER_NEAREST,
        )

    return color_bgr, depth_m


def _build_synthetic_K_from_depth(depth_m: np.ndarray) -> np.ndarray:
    """构造与 test_zeroshot 一致的简易 pinhole 内参。"""
    h, w = depth_m.shape
    fovy = np.pi / 4.0
    focal = h / (2.0 * np.tan(fovy / 2.0))
    K = np.array(
        [
            [focal, 0.0, w / 2.0],
            [0.0, focal, h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return K


def run_virtual_mode(args: argparse.Namespace) -> None:
    """虚拟数据最小回环：等价于原 test_zeroshot.py 的 main，并把中间结果都存盘。"""
    rgb_path = args.rgb
    depth_path = args.depth

    color_bgr, depth_m = _load_virtual_rgbd(rgb_path, depth_path)
    K = _build_synthetic_K_from_depth(depth_m)

    pick_text = args.pick_text or "抓起桌面上最显眼的一个物体"
    place_text = args.place_text or "把它放到桌面右下角的空白区域"

    # 确保输出目录存在，并把数值类基础数据存盘
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "virtual_depth_m.npy"), depth_m)
    np.savetxt(os.path.join(OUTPUT_DIR, "virtual_K.txt"), K)

    # 组件（与 test_zeroshot 一致）
    vlm = VLMClient()
    sam = SAMSegmenter()
    grasp_estimator = GraspEstimator()

    print(">>> [virtual] 开始 VLM 定位抓取/放置区域 ...")
    pick_res = vlm.locate_bbox(pick_text, color_bgr)
    place_res = vlm.locate_bbox(place_text, color_bgr)

    pick_coord = pick_res.get("coordinates") or {}
    place_coord = place_res.get("coordinates") or {}

    pick_bbox = pick_coord.get("bbox")
    place_bbox = place_coord.get("bbox")

    if not pick_bbox or len(pick_bbox) != 4:
        raise SystemExit(f"VLM 未返回有效抓取 bbox: {pick_res}")
    if not place_bbox or len(place_bbox) != 4:
        raise SystemExit(f"VLM 未返回有效放置 bbox: {place_res}")

    # 放置 bbox 中心像素 -> 3D 点（相机系）
    px1, py1, px2, py2 = map(int, place_bbox)
    place_u = (px1 + px2) / 2.0
    place_v = (py1 + py2) / 2.0
    place_point_cam = backproject_pixel_to_3d(place_u, place_v, depth_m, K)
    if place_point_cam is None:
        raise SystemExit("放置点像素深度无效，无法计算 3D 坐标。")

    # 可视化 VLM bbox
    visualize_vlm_bboxes(color_bgr, pick_bbox, place_bbox)

    print(">>> [virtual] 开始 SAM 分割抓取目标 ...")
    pick_center_uv, pick_mask = sam.segment_from_bbox(
        color_bgr, tuple(map(int, pick_bbox))
    )
    visualize_sam_mask(color_bgr, pick_mask)

    print(">>> [virtual] 开始 GraspNet 抓取估计 ...")
    gg_best, cloud_o3d = grasp_estimator.estimate_best_grasp_with_cloud(
        color_bgr, depth_m, K, pick_mask
    )
    T_cam_grasp = se3_from_grasp_in_cam(gg_best)

    # 基于抓取姿态的旋转，构造放置末端位姿（相机系）
    T_cam_place = np.eye(4, dtype=np.float32)
    T_cam_place[:3, :3] = T_cam_grasp[:3, :3]
    T_cam_place[:3, 3] = place_point_cam

    # Open3D 模式可视化点云 + 抓取位姿（完整夹爪）+ 放置末端位姿坐标系
    H_c, W_c, _ = color_bgr.shape
    visualize_cloud_and_grasp_o3d(
        cloud_o3d,
        gg_best,
        T_cam_place=T_cam_place,
        image_size=(W_c, H_c),
        K=K,
    )

    # 关键结果也存为文本，方便之后复现
    np.savetxt(
        os.path.join(OUTPUT_DIR, "virtual_grasp_pose_cam.txt"),
        T_cam_grasp,
    )
    np.savetxt(
        os.path.join(OUTPUT_DIR, "virtual_place_pose_cam.txt"),
        T_cam_place,
    )

    print("\n=== [virtual] 关键输出 ===")
    print("VLM 抓取回应:", pick_res.get("response"))
    print("VLM 放置回应:", place_res.get("response"))
    print("抓取 bbox:", pick_bbox)
    print("抓取中心像素:", pick_center_uv)
    print("抓取位姿 (cam frame) 4x4:\n", T_cam_grasp)
    print("放置 bbox:", place_bbox)
    print("放置中心像素:", (place_u, place_v))
    print("放置末端位姿 (cam frame) 4x4:\n", T_cam_place)


def run_online_mode(args: argparse.Namespace) -> None:
    """
    实机 RGBD + zeroshot：
      - 通过 UpperClient.capture() 获取一帧 (color_bgr, depth_m, K)
      - 假设已知 T_base_cam（可配置，默认单位=米的 4x4 txt 文件 / 或单位阵）
      - 参考 test_zeroshot 的流程，把中间可视化和关键结果都存盘
    """
    client = UpperClient(host=args.host, port=args.port)

    # 1. 获取一帧 RGBD + K
    print(f">>> [online] 连接下位机 {args.host}:{args.port} 并请求一帧 RGBD ...")
    color_bgr, depth_m, K = client.capture()
    print(
        f"[online] 收到彩图 {color_bgr.shape[1]}x{color_bgr.shape[0]} "
        f"深度 {depth_m.shape[1]}x{depth_m.shape[0]}"
    )

    # 保存数值类基础数据（图像可视化仍只用 debug_* 系列）
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "online_depth_m.npy"), depth_m)
    np.savetxt(os.path.join(OUTPUT_DIR, "online_K.txt"), K)

    pick_text = args.pick_text or "抓起桌面上最显眼的一个物体"
    place_text = args.place_text or "把它放到桌面右下角的空白区域"

    # 2. 构造在线执行所需的初始位姿（相机系）
    init_xyz_mm = np.array([-28.12, -200, 371.47], dtype=np.float32)
    init_rpy_deg = (0, 0, -98.87)
    T_cam_init = np.eye(4, dtype=np.float32)
    T_cam_init[:3, :3] = _euler_zyx_deg_to_matrix(*init_rpy_deg)
    T_cam_init[:3, 3] = init_xyz_mm / 1000.0

    # 3. 使用 ZeroShotPlanner 一次性完成规划 +（可选）动作序列生成
    planner = ZeroShotPlanner()
    print(">>> [online] 使用 ZeroShotPlanner 进行 zeroshot 抓取/放置规划 ...")
    result = planner.plan(
        color_bgr=color_bgr,
        depth=depth_m,
        K=K.astype(np.float32),
        pick_text=pick_text,
        place_text=place_text,
        return_motion=True,
        T_cam_init=T_cam_init,
        grip_open=50.0,
    )

    T_cam_grasp = result["grasp_pose_cam"]
    T_cam_place = result["place_pose_cam"]
    pick_bbox = result["pick_bbox"]
    place_bbox = result["place_bbox"]
    pick_center_uv = result["pick_center_uv"]
    place_u, place_v = result["place_center_uv"]
    place_point_cam = result["place_point_cam"]
    pick_mask = result["pick_mask"]
    cloud_o3d = result["cloud_o3d"]
    gg_best = result["grasp_group"]
    pick_res_text = result.get("vlm_pick_response")
    place_res_text = result.get("vlm_place_response")
    steps = result["motion_steps"]

    # 可视化 VLM bbox
    visualize_vlm_bboxes(color_bgr, pick_bbox, place_bbox)

    # 可视化 SAM mask
    visualize_sam_mask(color_bgr, pick_mask)

    # Open3D 模式可视化点云 + 抓取位姿（完整夹爪）+ 放置末端位姿坐标系
    H_c, W_c, _ = color_bgr.shape
    visualize_cloud_and_grasp_o3d(
        cloud_o3d,
        gg_best,
        T_cam_place=T_cam_place,
        image_size=(W_c, H_c),
        K=K.astype(np.float32),
        motion_steps=steps,
    )

    # 关键结果存盘（相机系下的 4x4 齐次矩阵）
    np.savetxt(
        os.path.join(OUTPUT_DIR, "online_grasp_pose_cam.txt"),
        T_cam_grasp,
    )
    np.savetxt(
        os.path.join(OUTPUT_DIR, "online_place_pose_cam.txt"),
        T_cam_place,
    )

    # 同时在终端打印一份“相机系下”的抓取位姿（更方便你对比下位机 log）
    t_cam = T_cam_grasp[:3, 3].astype(float)
    R_cam_raw = T_cam_grasp[:3, :3].astype(float)
    R_cam_corr = (_R_GRASPNET_TO_CAM @ R_cam_raw).astype(float)
    rx_deg_dbg, ry_deg_dbg, rz_deg_dbg = _rotation_matrix_to_euler_zyx_deg(R_cam_corr)
    x_mm_dbg, y_mm_dbg, z_mm_dbg = (t_cam * 1000.0).tolist()

    print("\n=== [online] 关键输出（仅计算，不发送到机械臂） ===")
    print("VLM 抓取回应:", pick_res_text)
    print("VLM 放置回应:", place_res_text)
    print("抓取 bbox:", pick_bbox)
    print("抓取中心像素:", pick_center_uv)
    print(
        "抓取位姿 (cam frame, mm/deg): "
        f"cam_xyz_mm=({x_mm_dbg:.3f}, {y_mm_dbg:.3f}, {z_mm_dbg:.3f}), "
        f"cam_rpy_deg=({rx_deg_dbg:.3f}, {ry_deg_dbg:.3f}, {rz_deg_dbg:.3f})"
    )
    print("放置 bbox:", place_bbox)
    print("放置中心像素:", (place_u, place_v))
    print("放置末端位姿 (cam frame) 4x4:\n", T_cam_place)

    # 等你检查 debug_*.png / 终端输出后，再决定是否发送抓取位姿到下位机
    try:
        user_in = input(
            "\n[online] 已生成 debug 图像和抓取位姿。"
            "按回车直接将“相机系抓取位姿”发送到下位机执行；"
            "输入 n 后回车则跳过本次发送: "
        ).strip().lower()
    except EOFError:
        user_in = "n"

    if user_in == "":
        # 从相机系的齐次矩阵中取出平移与旋转，并用 _R_GRASPNET_TO_CAM
        # 把 GraspNet 坐标系统一对齐到“正常的相机末端坐标系”。
        grip_open = 50.0

        # 由于在生成 debug 图和人工审核过程中可能间隔较久，
        # 下位机有可能已经主动关闭了之前用于 CAPTURE 的 TCP 连接。
        # 这里显式重连一次，使行为与单独运行 upper_client.py pose ... 一致。
        client.close()
        client = UpperClient(host=args.host, port=args.port)

        # 执行统一的序列；initial_grip=grip_open 与原逻辑一致
        execute_motion_sequence(client, steps, initial_grip=grip_open)
    else:
        print("[online] 本次已选择不发送抓取位姿给下位机。")

    client.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="zeroshot 统一入口：虚拟数据测试 / 实机 RGBD 抓取规划"
    )
    p.add_argument(
        "--mode",
        choices=["virtual", "online"],
        required=True,
        help="选择运行模式：virtual=虚拟RGBD最小回环；online=实机RGBD zeroshot 抓取规划",
    )

    # 虚拟模式参数
    p.add_argument(
        "--rgb",
        default="rgb.jpg",
        help="[virtual] RGB 图路径（默认当前目录 rgb.jpg）",
    )
    p.add_argument(
        "--depth",
        default="depth.png",
        help="[virtual] 深度图路径（默认当前目录 depth.png，单位 mm 的 16-bit PNG）",
    )

    # 通用指令文本
    p.add_argument(
        "--pick_text",
        default=None,
        help="抓取指令文本，默认“抓起桌面上最显眼的一个物体”",
    )
    p.add_argument(
        "--place_text",
        default=None,
        help="放置指令文本，默认“把它放到桌面右下角的空白区域”",
    )

    # online 模式相关
    p.add_argument("--host", default="10.5.23.176", help="[online] 下位机 IP")
    p.add_argument("--port", type=int, default=8888, help="[online] 下位机端口")
    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.mode == "virtual":
        run_virtual_mode(args)
    elif args.mode == "online":
        run_online_mode(args)
    else:
        raise SystemExit(f"未知 mode: {args.mode}")


if __name__ == "__main__":
    main()

