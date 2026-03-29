import os

import cv2
import numpy as np
import open3d as o3d

from zeroshot_pipeline import (
    ZeroShotPlanner,
    VLMClient,
    SAMSegmenter,
    GraspEstimator,
    se3_from_grasp_in_cam,
    backproject_pixel_to_3d,
    MotionStep,
)


OUTPUT_DIR = "outputs"


def visualize_vlm_bboxes(
    color_bgr: np.ndarray,
    pick_bbox,
    place_bbox,
    out_pick: str = os.path.join(OUTPUT_DIR, "debug_vlm_pick_bbox.png"),
    out_place: str = os.path.join(OUTPUT_DIR, "debug_vlm_place_bbox.png"),
) -> None:
    """在 RGB 图像上画出 VLM 返回的抓取/放置 bbox 并保存。"""
    img_pick = color_bgr.copy()
    x1, y1, x2, y2 = map(int, pick_bbox)
    cv2.rectangle(img_pick, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(out_pick, img_pick)

    img_place = color_bgr.copy()
    px1, py1, px2, py2 = map(int, place_bbox)
    cv2.rectangle(img_place, (px1, py1), (px2, py2), (0, 255, 0), 2)
    cv2.imwrite(out_place, img_place)


def visualize_sam_mask(
    color_bgr: np.ndarray,
    mask: np.ndarray,
    out_mask: str = os.path.join(OUTPUT_DIR, "debug_sam_mask.png"),
    out_mask_only: str = os.path.join(OUTPUT_DIR, "debug_sam_mask_only.png"),
) -> None:
    """
    将 SAM mask 清晰地可视化：
      - 保存一张仅有 mask 的黑白图（方便看分割轮廓）
      - 保存一张在 RGB 上用高对比颜色+边界线叠加的图
    """
    if mask.dtype != np.uint8:
        mask_u8 = (mask > 0).astype(np.uint8) * 255
    else:
        mask_u8 = mask

    # 1) 纯 mask 图（黑底白前景）
    cv2.imwrite(out_mask_only, mask_u8)

    # 2) 在 RGB 上叠加：将前景区域染成亮色，并画出边界
    overlay = color_bgr.copy()
    # 前景区域全部涂成亮绿色，确保与桌面有明显对比
    overlay[mask_u8 > 0] = (0, 255, 0)

    # 找轮廓并画粗边界
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=3)

    # 再和原图做一次 alpha 融合，保留一点原始纹理
    blended = cv2.addWeighted(color_bgr, 0.4, overlay, 0.6, 0)
    cv2.imwrite(out_mask, blended)


def visualize_cloud_and_grasp_o3d(
    cloud_o3d,
    grasp_group,
    T_cam_place=None,
    image_size=None,
    K=None,
    motion_steps=None,
) -> None:
    """
    使用 Open3D 实时可视化点云和 GraspNet 输出的抓取位姿（完整夹爪几何体），
    风格与 /data2/Project/Arm/ycliu/VLM_Grasp_Interactive 中的实现保持一致。
    """
    if len(grasp_group) == 0:
        print("[DEBUG] GraspGroup 为空，跳过可视化。")
        return

    geoms = grasp_group.to_open3d_geometry_list()
    # 为 place 末端位姿创建一个明显的几何体（如果给定）
    place_frame = None
    place_marker = None
    if T_cam_place is not None:
        place_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        place_frame.transform(T_cam_place)
        # 再加一个红色小球，方便一眼分辨 place 位置
        # 半径调小一点，避免在图里显得过大
        place_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        place_marker.paint_uniform_color([1.0, 0.0, 0.0])
        place_marker.transform(T_cam_place)
    if not cloud_o3d.has_colors():
        cloud_o3d.paint_uniform_color([0.7, 0.7, 0.7])

    # 额外：把 motion_steps 的每个 step 位姿也画出来（坐标系 + 轨迹折线）
    step_frames = []
    step_markers = []
    traj_line = None
    try:
        Ts = []
        if motion_steps is not None:
            if isinstance(motion_steps, list) and len(motion_steps) > 0:
                if isinstance(motion_steps[0], MotionStep):
                    Ts = [s.T_cam for s in motion_steps]
                else:
                    Ts = motion_steps
        pts = []
        for i, T in enumerate(Ts):
            Tm = np.asarray(T, dtype=np.float32)
            if Tm.shape != (4, 4):
                continue
            # 坐标系画小一点，避免遮挡
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            frame.transform(Tm)
            step_frames.append(frame)

            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            # 使用蓝->紫的渐变，便于看顺序
            t = 0.0 if len(Ts) <= 1 else float(i) / float(len(Ts) - 1)
            marker.paint_uniform_color([0.2 + 0.6 * t, 0.2, 1.0 - 0.7 * t])
            marker.transform(Tm)
            step_markers.append(marker)
            pts.append(Tm[:3, 3].reshape(3))

        if len(pts) >= 2:
            pts_np = np.stack(pts, axis=0)
            lines = [[i, i + 1] for i in range(len(pts) - 1)]
            colors = [[0.1, 0.1, 0.9] for _ in lines]
            traj_line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(pts_np.astype(np.float64)),
                lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
            )
            traj_line.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    except Exception as e:
        print(f"[WARN] 轨迹 steps 可视化构建失败: {e}")

    # 1) 如果有显示环境，正常弹 Open3D 窗口（可选，不需要可以删掉这一段）
    try:
        print(
            "[DEBUG] 打开 Open3D 窗口显示点云 + 抓取夹爪几何体 + 放置末端位姿（按 q 关闭窗口）"
        )
        draw_geoms = [cloud_o3d, *geoms]
        if place_frame is not None:
            draw_geoms.append(place_frame)
        if place_marker is not None:
            draw_geoms.append(place_marker)
        if step_frames:
            draw_geoms.extend(step_frames)
        if step_markers:
            draw_geoms.extend(step_markers)
        if traj_line is not None:
            draw_geoms.append(traj_line)
        o3d.visualization.draw_geometries(draw_geoms)
    except Exception as e:
        print(f"[WARN] Open3D 窗口显示失败（可能是无显示环境）: {e}")

        # 2) 使用 Open3D 的 OffscreenRenderer 渲染到图片（俯视图）
    try:
        from open3d.visualization import rendering

        # 如果给了原始图像分辨率，就用同样的分辨率渲染，便于对比
        if image_size is not None:
            width, height = image_size
        else:
            width, height = 800, 600
        renderer = rendering.OffscreenRenderer(width, height)

        # 背景色设置为白色，方便看
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

        # 点云材质（简单点：统一灰色）
        cloud_material = rendering.MaterialRecord()
        cloud_material.shader = "defaultUnlit"
        cloud_material.point_size = 1.0

        renderer.scene.add_geometry(
            "cloud", cloud_o3d, cloud_material
        )

        # 抓取几何体统一用默认材质加进来
        grasp_material = rendering.MaterialRecord()
        grasp_material.shader = "defaultUnlit"
        for i, g in enumerate(geoms):
            renderer.scene.add_geometry(
                f"grasp_{i}", g, grasp_material
            )

        # 放置末端位姿的坐标系和标记球也加入场景（如果有）
        if place_frame is not None:
            renderer.scene.add_geometry("place_pose_frame", place_frame, grasp_material)
        if place_marker is not None:
            renderer.scene.add_geometry("place_pose_marker", place_marker, grasp_material)

        # motion_steps 的位姿坐标系/轨迹
        for i, f in enumerate(step_frames):
            renderer.scene.add_geometry(f"step_frame_{i}", f, grasp_material)
        for i, m in enumerate(step_markers):
            renderer.scene.add_geometry(f"step_marker_{i}", m, grasp_material)
        if traj_line is not None:
            renderer.scene.add_geometry("traj_line", traj_line, grasp_material)

        # 相机设置：
        # - 如果提供了相机内参 K，则使用与原始 RGB 相机一致的 pinhole 相机模型，
        #   使渲染结果与 RGB/SAM 图的视角和像素坐标对齐。
        # - 否则退回到基于点云包围盒的简易视角。
        if K is not None:
            intrinsic = K.astype(np.float64)
            extrinsic = np.eye(4, dtype=np.float64)
            renderer.setup_camera(intrinsic, extrinsic, width, height)

            # 先按照“原始相机坐标系视角”渲染一张图片，便于与 RGB 对齐查看
            img_cam = renderer.render_to_image()
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_path_cam = os.path.join(
                OUTPUT_DIR, "debug_grasp_open3d_cam_view.png"
            )
            o3d.io.write_image(out_path_cam, img_cam)
            print(
                "[DEBUG] 已将 Open3D 相机坐标系视角渲染结果保存为图片:"
                f" {out_path_cam}"
            )
        else:
            bbox = renderer.scene.bounding_box
            center = bbox.get_center()
            extent = bbox.get_extent().max()
            eye = center + [0, -extent, extent]  # 从斜上方看
            up = [0, 0, 1]
            renderer.setup_camera(60.0, center, eye, up)

        # 加一点环境光
        renderer.scene.scene.set_indirect_light_intensity(1.0)

        img_top = renderer.render_to_image()

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path_top = os.path.join(OUTPUT_DIR, "debug_grasp_open3d.png")
        o3d.io.write_image(out_path_top, img_top)
        print(f"[DEBUG] 已将 Open3D 俯视图渲染结果保存为图片: {out_path_top}")

        # 3) 再渲染一个正视图（侧面视角），便于观察高度/立体关系
        bbox = renderer.scene.bounding_box
        center = bbox.get_center()
        extent = bbox.get_extent().max()
        # 侧视：沿 +x 方向看过去，上方向仍然取 +z
        eye_side = center + [extent, 0, 0]
        up_side = [0, 0, 1]
        renderer.setup_camera(60.0, center, eye_side, up_side)

        img_side = renderer.render_to_image()
        out_path_side = os.path.join(OUTPUT_DIR, "debug_grasp_open3d_side.png")
        o3d.io.write_image(out_path_side, img_side)
        print(f"[DEBUG] 已将 Open3D 正视图渲染结果保存为图片: {out_path_side}")
    except Exception as e:
        print(
            "[ERROR] Open3D OffscreenRenderer 渲染失败，"
            "请确认 Open3D 编译时支持 OSMesa/EGL 等离线渲染后端。错误信息:"
        )
        print(e)


def main() -> None:
    """
    使用当前目录下的 rgb.png 和 depth.png 模拟机械臂看到的场景，
    按照 VLM -> SAM -> GraspNet 的完整流程推理，并将中间结果全部可视化：
      - VLM 定位的抓取/放置 bbox（叠加在 RGB 上）
      - SAM 分割掩码（叠加在 RGB 上）
      - GraspNet 估计的抓取位姿 + 3D 点云（matplotlib 3D 图）
    """
    # 统一输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rgb_path = "rgb.jpg"
    depth_path = "depth.png"

    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        raise SystemExit("当前目录下未找到 rgb.png 或 depth.png，请先运行 genrgbd.py 生成示例数据。")

    # 1. 读取 RGB 图（BGR 格式）
    color_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise SystemExit("读取 rgb.png 失败。")

    # 2. 读取深度图：depth.png 为 16-bit PNG，单位：毫米
    raw_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if raw_depth is None:
        raise SystemExit("读取 depth.png 失败。")

    if raw_depth.dtype != np.uint16:
        print(f"[警告] depth.png 非 16-bit，当前 dtype = {raw_depth.dtype}")

    depth_m = raw_depth.astype(np.float32) / 1000.0  # 毫米 -> 米

    # 2.1 如果 RGB 和深度分辨率不一致，则对深度图做插值/降采样，使其与 RGB 一致
    H_d, W_d = depth_m.shape
    H_c, W_c, _ = color_bgr.shape
    if (H_d, W_d) != (H_c, W_c):
        print(
            f"[INFO] RGB 分辨率为 {H_c}x{W_c}, 深度分辨率为 {H_d}x{W_d}, "
            "对深度图进行插值以对齐到 RGB 分辨率。"
        )
        # 对深度图使用最近邻插值以避免产生非物理值；如需更平滑可改为 INTER_LINEAR
        depth_m = cv2.resize(
            depth_m,
            (W_c, H_c),
            interpolation=cv2.INTER_NEAREST,
        )
        H_d, W_d = depth_m.shape

    # 3. 构造一个简单的 pinhole 相机内参（与 genrgbd.py 中设置保持一致）
    H, W = depth_m.shape
    fovy = np.pi / 4.0
    focal = H / (2.0 * np.tan(fovy / 2.0))
    K = np.array(
        [
            [focal, 0.0, W / 2.0],
            [0.0, focal, H / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # 4. 简单假设：相机坐标系和机械臂 base 坐标系重合（只为算法测试）
    T_base_cam = np.eye(4, dtype=np.float32)

    # 组件
    vlm = VLMClient()
    sam = SAMSegmenter()
    grasp_estimator = GraspEstimator()

    pick_text = "抓起桌面上最显眼的一个物体" # 抓起桌面上的墨镜
    place_text = "把它放到桌面右下角的空白区域"

    print(">>> 开始 VLM 定位抓取/放置区域 ...")
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

    # 计算放置区域中心像素，并回投到 3D，得到放置点在相机坐标系下的位置
    px1, py1, px2, py2 = map(int, place_bbox)
    place_u = (px1 + px2) / 2.0
    place_v = (py1 + py2) / 2.0
    place_point_cam = backproject_pixel_to_3d(place_u, place_v, depth_m, K)
    if place_point_cam is None:
        raise SystemExit("放置点像素深度无效，无法计算 3D 坐标。")

    # 可视化 VLM bbox
    visualize_vlm_bboxes(color_bgr, pick_bbox, place_bbox)

    print(">>> 开始 SAM 分割抓取目标 ...")
    pick_center_uv, pick_mask = sam.segment_from_bbox(
        color_bgr, tuple(map(int, pick_bbox))
    )
    visualize_sam_mask(color_bgr, pick_mask)

    print(">>> 开始 GraspNet 抓取估计 ...")
    gg_best, cloud_o3d = grasp_estimator.estimate_best_grasp_with_cloud(
        color_bgr, depth_m, K, pick_mask
    )
    T_cam_grasp = se3_from_grasp_in_cam(gg_best)
    T_base_grasp = T_base_cam @ T_cam_grasp

    # 基于抓取姿态的旋转，构造放置末端位姿（相机系）
    T_cam_place = np.eye(4, dtype=np.float32)
    T_cam_place[:3, :3] = T_cam_grasp[:3, :3]
    T_cam_place[:3, 3] = place_point_cam
    T_base_place = T_base_cam @ T_cam_place

    # Open3D 模式可视化点云 + 抓取位姿（完整夹爪）+ 放置末端位姿坐标系
    # 使用与原始 RGB 相同的分辨率渲染，便于和其他 debug 图片对齐
    visualize_cloud_and_grasp_o3d(
        cloud_o3d,
        gg_best,
        T_cam_place=T_cam_place,
        image_size=(W_c, H_c),
        K=K,
    )

    print("\n=== 关键输出 ===")
    print("VLM 抓取回应:", pick_res.get("response"))
    print("VLM 放置回应:", place_res.get("response"))
    print("抓取 bbox:", pick_bbox)
    print("抓取中心像素:", pick_center_uv)
    print("抓取位姿 (base frame) 4x4:\n", T_base_grasp)
    print("放置 bbox:", place_bbox)
    print("放置中心像素:", (place_u, place_v))
    print("放置末端位姿 (base frame) 4x4:\n", T_base_place)


if __name__ == "__main__":
    main()

