import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List

import cv2
import numpy as np
import torch
from openai import OpenAI
from ultralytics.models.sam import Predictor as SAMPredictor


"""
零样本抓取与放置规划管线（RGBD -> VLM+SAM -> 点云 -> GraspNet -> 6D抓取 & 放置位姿）

依赖（与 /data2/Project/Arm/ycliu/VLM_Grasp_Interactive 保持一致）:
  - graspnet-baseline（模型、数据集、utils）
  - logs/log_rs/checkpoint-rs.tar 预训练权重
  - ultralytics 的 SAM 模型权重 sam_b.pt
  - 一个可用的多模态 VLM（这里使用 Qwen OpenAI-compatible API）

注意：
  - 本文件只负责“计算”抓取位姿和放置终点位姿，不直接控制机械臂。
  - 你需要提供：RGB 图、深度图、相机内参 K、相机到机械臂基座的外参 T_base_cam。

运行前请先设置环境变量（至少包含你实际会用到的那一套）：
  - DASHSCOPE_API_KEY         : Qwen-VL（DashScope OpenAI-compatible）API Key（用于 VLM 定位 bbox）
  - DEEPSEEK_API_KEY          : DeepSeek API Key（用于 DeepSeek-R1 轨迹规划）
  - DEEPSEEK_BASE_URL         : DeepSeek OpenAI-compatible base_url（可选；默认 https://api.deepseek.com）

面向对象设计：
  - GraspEstimator : 封装 GraspNet 点云预处理与抓取姿态估计。
  - VLMClient      : 封装多模态大模型调用，根据指令在图像中返回 bbox。
  - SAMSegmenter   : 封装 SAM 分割，根据 bbox 或点生成 mask。
  - ZeroShotPlanner: 组合以上组件，输入 RGBD + 文本指令，输出抓取/放置位姿。
"""

os.environ["DASHSCOPE_API_KEY"] = ""
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com"

# =============================================================================
# GraspNet 相关：从参考工程复制并简化
# =============================================================================

ROOT_VLM_GRASP = "/data2/Project/Arm/ycliu/VLM_Grasp_Interactive"
GRASPNET_LOG_CKPT = os.path.join(
    ROOT_VLM_GRASP, "logs", "log_rs", "checkpoint-rs.tar"
)

# 把 graspnet-baseline 加入 sys.path
sys.path.append(os.path.join(ROOT_VLM_GRASP, "graspnet-baseline", "models"))
sys.path.append(os.path.join(ROOT_VLM_GRASP, "graspnet-baseline", "dataset"))
sys.path.append(os.path.join(ROOT_VLM_GRASP, "graspnet-baseline", "utils"))

from graspnet import GraspNet, pred_decode  # type: ignore  # noqa: E402
from collision_detector import (  # type: ignore  # noqa: E402
    ModelFreeCollisionDetector,
)
from data_utils import (  # type: ignore  # noqa: E402
    CameraInfo,
    create_point_cloud_from_depth_image,
)
from graspnetAPI import GraspGroup  # type: ignore  # noqa: E402


class GraspEstimator:
    """
    封装 GraspNet 相关逻辑：RGBD -> 点云 -> GraspNet -> 最优抓取姿态（相机系）。
    """

    def __init__(
        self,
        checkpoint_path: str = GRASPNET_LOG_CKPT,
        device: Optional[torch.device] = None,
        depth_max: float = 2.0,
        num_points: int = 5000,
        collision_thresh: float = 0.01,
        vertical_angle_deg: float = 30.0,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.depth_max = depth_max
        self.num_points = num_points
        self.collision_thresh = collision_thresh
        self.vertical_angle_rad = np.deg2rad(vertical_angle_deg)

        self._net: Optional[GraspNet] = None

    # ---------------- 内部工具 ----------------
    def _load_net(self) -> GraspNet:
        if self._net is not None:
            return self._net

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"GraspNet checkpoint not found: {self.checkpoint_path}"
            )

        net = GraspNet(
            input_feature_dim=0,
            num_view=300,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        net.to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()

        self._net = net
        return net

    def _rgbd_to_grasp_input(
        self,
        color_bgr: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        sam_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, Any], "object"]:
        """
        参考 grasp_process.get_and_process_data：
        根据 RGB、深度和 SAM mask 生成点云和 GraspNet 所需 end_points。
        """
        assert depth.ndim == 2, "depth 必须是单通道 (H, W)"

        h, w = depth.shape

        # 颜色归一化到 [0,1]
        color = color_bgr.astype(np.float32)
        if color.max() > 1.0:
            color /= 255.0

        # 处理 SAM mask：用“放大后的外接矩形”作为点云工作区，而不是只用精细 mask
        if sam_mask is None:
            workspace_mask = np.ones_like(depth, dtype=bool)
        else:
            m = sam_mask
            if m.ndim == 3:
                m = m[..., 0]
            if m.shape != depth.shape:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            m_bin = (m > 0).astype(np.uint8)

            # 计算 mask 的外接矩形，并在四周各扩展 margin 像素，防止点云过窄
            ys, xs = np.where(m_bin > 0)
            if len(xs) == 0 or len(ys) == 0:
                # 极端情况：mask 为空，就退化为全局点云
                workspace_mask = np.ones_like(depth, dtype=bool)
            else:
                # 外接矩形四周扩展一小圈：取 min(H,W) 的 3% 或至少 5 像素
                margin = max(int(0.01 * min(h, w)),2)
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                x1 = max(x1 - margin, 0)
                x2 = min(x2 + margin, w - 1)
                y1 = max(y1 - margin, 0)
                y2 = min(y2 + margin, h - 1)

                workspace_mask = np.zeros_like(depth, dtype=bool)
                workspace_mask[y1 : y2 + 1, x1 : x2 + 1] = True

        factor_depth = 1.0
        cam_info = CameraInfo(
            width=w,
            height=h,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            scale=factor_depth,
        )

        cloud = create_point_cloud_from_depth_image(depth, cam_info, organized=True)

        valid_mask = workspace_mask & (depth < self.depth_max)
        cloud_masked = cloud[valid_mask]
        color_masked = color[valid_mask]

        if len(cloud_masked) == 0:
            raise RuntimeError("点云为空，请检查深度图与 mask 是否正确。")

        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(
                len(cloud_masked), self.num_points - len(cloud_masked), replace=True
            )
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        import open3d as o3d  # 延迟导入

        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

        cloud_sampled_t = torch.from_numpy(
            cloud_sampled[np.newaxis].astype(np.float32)
        ).to(self.device)

        end_points: Dict[str, Any] = {
            "point_clouds": cloud_sampled_t,
            "cloud_colors": color_sampled,
        }
        return end_points, cloud_o3d

    # ---------------- 对外主接口 ----------------
    def estimate_best_grasp_with_cloud(
        self,
        color_bgr: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        sam_mask: Optional[np.ndarray],
    ) -> Tuple[GraspGroup, "object"]:
        """
        使用 GraspNet 对给定 mask 内的物体进行抓取姿态估计，
        返回只包含一个最佳抓取的 GraspGroup（相机坐标系）以及对应的点云 cloud_o3d。
        """
        net = self._load_net()
        end_points, cloud_o3d = self._rgbd_to_grasp_input(color_bgr, depth, K, sam_mask)

        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)

        gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())

        # 碰撞检测
        import open3d as o3d  # 延迟导入

        if self.collision_thresh > 0:
            voxel_size = 0.01
            mfcdetector = ModelFreeCollisionDetector(
                np.asarray(cloud_o3d.points), voxel_size=voxel_size
            )
            collision_mask = mfcdetector.detect(
                gg, approach_dist=0.05, collision_thresh=self.collision_thresh
            )
            gg = gg[~collision_mask]

        # NMS + score 排序
        gg.nms().sort_by_score()

        # 垂直约束
        all_grasps = list(gg)
        vertical = np.array([0, 0, 1])
        filtered = []
        for grasp in all_grasps:
            approach_dir = grasp.rotation_matrix[:, 0]
            cos_angle = np.dot(approach_dir, vertical)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle < self.vertical_angle_rad:
                filtered.append(grasp)
        if not filtered:
            filtered = all_grasps

        # 距离 & 综合得分（和参考实现一致）
        points = np.asarray(cloud_o3d.points)
        object_center = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)

        distances = []
        for grasp in filtered:
            grasp_center = grasp.translation
            distances.append(np.linalg.norm(grasp_center - object_center))

        grasp_with_distances = list(zip(filtered, distances))
        grasp_with_distances.sort(key=lambda x: x[1])

        max_distance = max(distances) if distances else 1.0
        grasp_with_scores = []
        for g, d in grasp_with_distances:
            distance_score = 1 - (d / max_distance)
            composite_score = g.score * 0.1 + distance_score * 0.9
            grasp_with_scores.append((g, composite_score))

        grasp_with_scores.sort(key=lambda x: x[1], reverse=True)
        best_grasp = grasp_with_scores[0][0]

        new_gg = GraspGroup()
        new_gg.add(best_grasp)
        return new_gg, cloud_o3d

    def estimate_best_grasp(
        self,
        color_bgr: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        sam_mask: Optional[np.ndarray],
    ) -> GraspGroup:
        """
        使用 GraspNet 对给定 mask 内的物体进行抓取姿态估计。
        返回只包含一个最佳抓取的 GraspGroup（相机坐标系）。
        """
        gg, _ = self.estimate_best_grasp_with_cloud(color_bgr, depth, K, sam_mask)
        return gg


# =============================================================================
# VLM + SAM：从参考工程提取并简化
# =============================================================================

import base64  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import textwrap  # noqa: E402

# export DASHSCOPE_API_KEY="sk-21a42456345b4b6da1df9d0a08a4396c"

class VLMClient:
    """
    封装多模态大模型调用逻辑：给定图像 + 文本指令，返回目标 bbox。
    """

    def __init__(
        self,
        api_key_env: str = "DASHSCOPE_API_KEY",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-vl-max",
        temperature: float = 0.1,
    ) -> None:
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self._client: Optional[OpenAI] = None

    @staticmethod
    def _encode_np_image_to_base64(image_bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", image_bgr)
        if not ok:
            raise ValueError("图像编码 JPEG 失败")
        return base64.b64encode(buf).decode("utf-8")

    @staticmethod
    def _build_system_prompt() -> str:
        return textwrap.dedent(
            """\
            你是一个精密机械臂视觉控制系统，具备多模态感知能力。
            任务：在图像中根据用户指令找到最相关的目标区域（物体或区域），并返回其 2D 边界框。

            输出格式必须严格如下：
            - 先给一段自然语言说明（可以用中文，解释你选择的目标），
            - 下一行开始返回一个 **纯 JSON 对象**，格式：
            {
              "name": "物体名称",
              "bbox": [左上角x, 左上角y, 右下角x, 右下角y]
            }

            要求：
            - JSON 必须单独从新的一行开始；
            - JSON 里不要有注释或额外文本；
            - 只允许字段 "name" 和 "bbox"；
            - bbox 坐标必须是整数像素坐标。
            """
        )

    @staticmethod
    def _extract_text_and_json(content: str) -> Dict[str, Any]:
        m = re.search(r"(\{.*\})", content, re.DOTALL)
        if m:
            js = m.group(1)
            try:
                coord = json.loads(js)
            except Exception:
                coord = {}
            natural = content[: m.start()].strip()
        else:
            natural = content.strip()
            coord = {}
        return {"response": natural, "coordinates": coord}

    def _build_client(self) -> OpenAI:
        """
        懒加载并缓存 OpenAI 兼容客户端，避免每次调用都重新创建连接对象。
        """
        if self._client is not None:
            return self._client

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"环境变量 {self.api_key_env} 未设置，无法调用 VLM 接口。"
            )

        self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    def locate_bbox(
        self, prompt: str, image_bgr: np.ndarray
    ) -> Dict[str, Any]:
        """
        使用 VLM（Qwen-VL）在图像中根据文本指令返回一个 bbox。

        返回：
          {
            "response": 自然语言说明,
            "coordinates": {"name": ..., "bbox": [x1,y1,x2,y2]} 或 {}
          }
        """
        client = self._build_client()
        system_prompt = self._build_system_prompt()

        img_b64 = self._encode_np_image_to_base64(image_bgr)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        completion = client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature
        )
        content = completion.choices[0].message.content
        return self._extract_text_and_json(content)


class DeepSeekR1TrajectoryPlanner:
    """
    使用 DeepSeek-R1（OpenAI-compatible API）基于 3D 点与任务文本规划抓取后的夹爪轨迹。
    约定：
      - 输入与输出均在相机坐标系下表达
      - 输出为一串 waypoints（SE3 或 position-only），用于上层进一步做 IK/控制
    """

    def __init__(
        self,
        api_key_env: str = "DEEPSEEK_API_KEY",
        base_url_env: str = "DEEPSEEK_BASE_URL",
        base_url_default: str = "https://api.deepseek.com",
        model: str = "deepseek-chat", # "deepseek-reasoner",
        temperature: float = 0.1,
        timeout_s: float = 60.0,
    ) -> None:
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self.base_url_default = base_url_default
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._client: Optional[OpenAI] = None

    def _build_client(self) -> OpenAI:
        if self._client is not None:
            return self._client
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"环境变量 {self.api_key_env} 未设置，无法调用 DeepSeek-R1 接口。"
            )
        base_url = os.environ.get(self.base_url_env) or self.base_url_default
        # 显式超时：避免网络/服务端阻塞导致 plan() 卡死
        # 注：OpenAI Python SDK 支持通过 http_client 注入超时配置。
        try:
            import httpx  # type: ignore

            http_client = httpx.Client(timeout=httpx.Timeout(self.timeout_s))
            self._client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        except Exception:
            # 退化：不注入 httpx（仍可能无超时）；但尽量保持兼容不崩
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        return self._client

    @staticmethod
    def _build_system_prompt() -> str:
        return textwrap.dedent(
            """\
            你是机器人运动规划器。你会根据：
            - 抓取物体的局部 3D 结构采样点（相机系）
            - 放置目标位置的 3D 点（相机系）
            - 任务指令文本
            规划“抓取后夹爪”的运动轨迹 waypoints，**只需规划位置 pos=[x,y,z]，姿态保持与抓取时一致**。

            坐标系与“上/下”方向说明（非常重要）：
            - 所有 pos/点云/位姿都在同一个相机坐标系 cam 下表达；
            - cam 为左手系，(x, y, z) = (左, 下, 外)；
            - 在本系统中，**z 轴负方向是“上”**，z 轴正方向是“向下靠近桌面/地面”；
            - 因此：如果要让夹爪“抬高离开桌面”，请让 z 变得更小（更负）；要“降低靠近桌面”，请让 z 变大（更正）。

            机械臂的姿态（旋转）在整个轨迹中保持与抓取时一致，你**不需要推理或规划 quat**；
            即使你在 JSON 中返回了 quat 字段，系统也会忽略这些旋转，只使用你给出的 pos。

           
            安全与约束：
            - 尽量避免与物体/桌面碰撞：先抬高再平移再下降（若合理）
            - 轨迹要平滑，waypoint 数量 5~12 个即可
            - 输出只允许一个 JSON 对象，必须从新的一行开始，且不包含注释/额外文本

            输出 JSON 格式（严格遵守字段名）：
            {
              "frame": "cam",
              "waypoints": [
                {
                  "pos": [x, y, z],
                  "quat": [qx, qy, qz, qw],
                  "grip": 0.0,
                  "note": "可选简短说明"
                }
              ]
            }

            说明：
            - pos 单位：米；quat 为单位四元数
            - grip 表示夹爪开合量（抓取后保持闭合：0.0）
            - 如果你无法可靠给出姿态，请让 quat = [0,0,0,1]（保持默认姿态），但仍需给出 waypoints。
            """
        )

    @staticmethod
    def _extract_first_json_obj(text: str) -> Dict[str, Any]:
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if not m:
            return {}
        js = m.group(1)
        try:
            return json.loads(js)
        except Exception:
            return {}

    def plan_trajectory(
        self,
        *,
        task_pick: str,
        task_place: str,
        pick_points_cam: np.ndarray,
        place_point_cam: np.ndarray,
        T_cam_grasp: np.ndarray,
        T_cam_place: np.ndarray,
    ) -> Dict[str, Any]:
        """
        返回 DeepSeek-R1 规划的轨迹 JSON（解析后的 dict）。
        """
        client = self._build_client()
        system_prompt = self._build_system_prompt()

        payload = {
            "task": {"pick_text": task_pick, "place_text": task_place},
            "inputs": {
                "pick_points_cam_m": np.asarray(pick_points_cam, dtype=float).tolist(),
                "place_point_cam_m": np.asarray(place_point_cam, dtype=float).tolist(),
                "T_cam_grasp": np.asarray(T_cam_grasp, dtype=float).tolist(),
                "T_cam_place": np.asarray(T_cam_place, dtype=float).tolist(),
            },
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "请根据以下输入规划抓取后夹爪轨迹，按要求输出 JSON：\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            },
        ]

        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        content = completion.choices[0].message.content or ""
        return self._extract_first_json_obj(content)


class SAMSegmenter:
    """
    封装 SAM 分割：根据 bbox 在图像上生成 mask 和中心坐标。
    """

    def __init__(
        self,
        sam_weight_path: str = os.path.join(ROOT_VLM_GRASP, "sam_b.pt"),
        conf: float = 0.25,
    ) -> None:
        self.sam_weight_path = sam_weight_path
        self.conf = conf
        self._predictor: Optional[SAMPredictor] = None

    def _get_predictor(self) -> SAMPredictor:
        if self._predictor is not None:
            return self._predictor

        overrides = dict(
            task="segment",
            mode="predict",
            model=self.sam_weight_path,
            conf=self.conf,
            save=False,
        )
        self._predictor = SAMPredictor(overrides=overrides)
        return self._predictor

    def segment_from_bbox(
        self, image_bgr: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[Tuple[int, int], np.ndarray]:
        """
        使用 SAM 对给定 bbox 内目标进行分割。
        返回 (mask_center_uv, mask_binary_uint8)
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor = self._get_predictor()
        predictor.set_image(image_rgb)

        x1, y1, x2, y2 = bbox
        results = predictor(bboxes=[[x1, y1, x2, y2]])
        if not results or not results[0].masks:
            raise RuntimeError("SAM 未返回有效 mask")

        mask = results[0].masks.data[0].cpu().numpy()
        mask = (mask > 0).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise RuntimeError("SAM 分割结果轮廓为空")
        M = cv2.moments(contours[0])
        if M["m00"] == 0:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
        else:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
        return center, mask


# =============================================================================
# 几何工具：像素 <-> 3D
# =============================================================================

def backproject_pixel_to_3d(
    u: float, v: float, depth: np.ndarray, K: np.ndarray
) -> Optional[np.ndarray]:
    """
    从像素坐标 (u,v) 与深度图中取深度值，回投到相机坐标系下的 3D 点。
    如果该点深度无效，返回 None。
    """
    h, w = depth.shape
    u_i = int(round(u))
    v_i = int(round(v))
    if u_i < 0 or u_i >= w or v_i < 0 or v_i >= h:
        return None
    z = float(depth[v_i, u_i])
    if not np.isfinite(z) or z <= 0:
        return None

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def sample_mask_pixels(
    mask: np.ndarray,
    *,
    num_samples: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    从二值 mask 中随机采样像素点，返回 shape=(N,2) 的 (u,v) 浮点坐标（像素系）。
    """
    if rng is None:
        rng = np.random.default_rng()
    m = mask
    if m.ndim == 3:
        m = m[..., 0]
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    idx = rng.choice(len(xs), size=min(num_samples, len(xs)), replace=False)
    uv = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    return uv


def backproject_uvs_to_3d_points(
    uvs: np.ndarray, depth: np.ndarray, K: np.ndarray
) -> np.ndarray:
    """
    将一组 (u,v) 回投到相机系 3D 点，过滤掉无效深度，返回 shape=(M,3)。
    """
    pts: List[np.ndarray] = []
    for u, v in np.asarray(uvs, dtype=float):
        p = backproject_pixel_to_3d(float(u), float(v), depth, K)
        if p is not None:
            pts.append(p)
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(pts, axis=0).astype(np.float32)


def _quat_to_rotmat_xyzw(quat: np.ndarray) -> np.ndarray:
    """
    将四元数 (qx,qy,qz,qw) 转换为旋转矩阵。
    若 quat 非法或近似单位阵，返回单位阵。
    """
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = q.tolist()
    n = x * x + y * y + z * z + w * w
    if not np.isfinite(n) or n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    R = np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )
    return R


def build_motion_steps_from_llm_traj(
    llm_traj: Dict[str, Any],
    *,
    T_cam_grasp: np.ndarray,
    default_grip_after_grasp: float = 0.0,
) -> List["MotionStep"]:
    """
    将 LLM 输出的轨迹 waypoints 转成 MotionStep 列表（每一步都带在线交互 prompt）。
    - 轨迹默认表达在相机系 frame="cam"
    - 若 waypoint 未提供有效 quat，则复用抓取位姿的旋转
    """
    if not isinstance(llm_traj, dict):
        return []
    if llm_traj.get("frame") not in (None, "cam"):
        return []
    wps = llm_traj.get("waypoints")
    if not isinstance(wps, list) or len(wps) == 0:
        return []

    steps: List[MotionStep] = []
    R_fallback = np.asarray(T_cam_grasp[:3, :3], dtype=np.float32)

    for i, wp in enumerate(wps):
        if not isinstance(wp, dict):
            continue
        pos = wp.get("pos")
        if (
            not isinstance(pos, list)
            or len(pos) != 3
            or not all(np.isfinite(float(x)) for x in pos)
        ):
            continue
        # 轨迹规划中**不使用 LLM 提供的 quat**：
        # 姿态在整个轨迹中保持与抓取时一致，避免 LLM 对旋转方向产生误解。
        R = R_fallback

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(pos, dtype=np.float32)

        grip = wp.get("grip", default_grip_after_grasp)
        try:
            grip_f = float(grip)
        except Exception:
            grip_f = float(default_grip_after_grasp)

        note = wp.get("note")
        note_s = str(note) if note is not None else ""
        desc = f"LLM 轨迹点 {i+1}/{len(wps)}" + (f" - {note_s}" if note_s else "")
        prompt = (
            "\n[online] 是否执行该 LLM 轨迹点？"
            "按回车执行本步；输入 n 后回车则跳过本步: "
        )
        steps.append(
            MotionStep(
                T_cam=T,
                grip=grip_f,
                desc=desc,
                prompt=prompt,
                keep_grip=True,
            )
        )
    return steps


def se3_from_grasp_in_cam(gg: GraspGroup) -> np.ndarray:
    """
    将 GraspGroup 中第一个抓取转换为 4x4 SE(3) 齐次矩阵（相机坐标系）。
    """
    g = gg[0]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = g.rotation_matrix
    T[:3, 3] = g.translation
    return T


# =============================================================================
# 顶层接口：zeroshot 规划函数
# =============================================================================

class ZeroShotPlanner:
    """
    零样本抓取与放置规划器：
      - VLM + SAM: 找到需要抓的物体和放置区域的 2D 区域并分割出目标 mask；
      - GraspNet: 在 mask 内估计 6D 抓取位姿；
      - 几何变换: 从相机坐标系转换到机械臂 base 坐标系，并根据放置区域生成终点位姿。
    """

    def __init__(
        self,
        vlm_client: Optional[VLMClient] = None,
        sam_segmenter: Optional[SAMSegmenter] = None,
        grasp_estimator: Optional[GraspEstimator] = None,
        traj_planner: Optional[DeepSeekR1TrajectoryPlanner] = None,
    ) -> None:
        self.vlm_client = vlm_client or VLMClient()
        self.sam_segmenter = sam_segmenter or SAMSegmenter()
        self.grasp_estimator = grasp_estimator or GraspEstimator()
        self.traj_planner = traj_planner

    def plan(
        self,
        color_bgr: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        pick_text: str,
        place_text: str,
        *,
        return_motion: bool = False,
        return_llm_traj: bool = True,
        T_cam_init: Optional[np.ndarray] = None,
        grip_open: float = 50.0,
    ) -> Dict[str, Any]:
        """
        零样本抓取与放置规划主函数（面向对象版本）。

        输入：
          - color_bgr: 当前场景 RGB 图 (H,W,3), BGR 格式（OpenCV）
          - depth: 对应深度图 (H,W), 单位: 米
          - K: 3x3 相机内参矩阵
          - pick_text: 文本指令，说明要抓取哪个物体（如：“抓起桌上的红色杯子”）
          - place_text: 文本指令，说明要放到什么位置（如：“放到桌子右侧空白区域”）

        返回字典：
          {
            "grasp_pose_cam": 4x4 抓取位姿 (cam frame),
            "place_pose_cam": 4x4 放置终点位姿 (cam frame),
            "pick_bbox": [x1,y1,x2,y2],
            "pick_center_uv": (u,v),
            "place_bbox": [x1,y1,x2,y2],
            "place_center_uv": (u,v),
            "place_point_cam": (3,)  # 相机系下的放置点 3D 坐标
          }
        """
        # 1) VLM 选择抓取目标 bbox
        pick_res = self.vlm_client.locate_bbox(pick_text, color_bgr)
        pick_coord = pick_res.get("coordinates") or {}
        pick_bbox = pick_coord.get("bbox")
        if not pick_bbox or len(pick_bbox) != 4:
            raise RuntimeError(f"VLM 未返回有效抓取 bbox: {pick_res}")
        x1, y1, x2, y2 = map(int, pick_bbox)

        # 2) SAM 细化分割 -> mask & 目标中心像素
        pick_center_uv, pick_mask = self.sam_segmenter.segment_from_bbox(
            color_bgr, (x1, y1, x2, y2)
        )

        # 3) GraspNet 估计最佳抓取位姿（相机系）以及对应点云
        gg_best, cloud_o3d = self.grasp_estimator.estimate_best_grasp_with_cloud(
            color_bgr, depth, K, pick_mask
        )
        T_cam_grasp = se3_from_grasp_in_cam(gg_best)

        # 4) 转到 base 坐标系
        # 5) VLM 选择放置区域 bbox
        place_res = self.vlm_client.locate_bbox(place_text, color_bgr)
        place_coord = place_res.get("coordinates") or {}
        place_bbox = place_coord.get("bbox")
        if not place_bbox or len(place_bbox) != 4:
            raise RuntimeError(f"VLM 未返回有效放置 bbox: {place_res}")
        px1, py1, px2, py2 = map(int, place_bbox)
        place_u = (px1 + px2) / 2.0
        place_v = (py1 + py2) / 2.0

        # 6) 回投到 3D，得到放置点在相机系下的坐标
        place_point_cam = backproject_pixel_to_3d(place_u, place_v, depth, K)
        if place_point_cam is None:
            raise RuntimeError("放置点像素深度无效，无法计算 3D 坐标。")

        # 7) 构造放置位姿（简单做法：复用抓取的姿态，只改变位置）
        T_cam_place = np.eye(4, dtype=np.float32)
        T_cam_place[:3, :3] = T_cam_grasp[:3, :3]
        T_cam_place[:3, 3] = place_point_cam

        # 7.5) LLM 轨迹规划：从 SAM mask 采样 5 点 -> 投影到深度得到 3D 点，连同放置 3D 点输入 DeepSeek-R1
        llm_traj: Optional[Dict[str, Any]] = None
        pick_points_cam = np.zeros((0, 3), dtype=np.float32)
        if return_llm_traj:
            try:
                t0 = time.perf_counter()
                print("\n[LLM-TRAJ] 开始规划抓取后轨迹 ...")
                print(
                    "[LLM-TRAJ] env: "
                    f"DEEPSEEK_API_KEY={'SET' if os.environ.get('DEEPSEEK_API_KEY') else 'MISSING'}, "
                    f"DEEPSEEK_BASE_URL={os.environ.get('DEEPSEEK_BASE_URL') or '(default)'}"
                )
                print(
                    f"[LLM-TRAJ] inputs: depth_shape={tuple(depth.shape)}, "
                    f"K_shape={tuple(K.shape)}, pick_mask_shape={tuple(pick_mask.shape)}"
                )

                # 先从 mask 采样像素，再回投到 3D；如果有效点不足，增加采样次数做补齐
                rng = np.random.default_rng()
                uv_all: List[np.ndarray] = []
                for _ in range(3):  # 最多三轮扩充候选点
                    uv = sample_mask_pixels(pick_mask, num_samples=10, rng=rng)
                    if uv.shape[0] > 0:
                        uv_all.append(uv)
                if uv_all:
                    uv_cat = np.concatenate(uv_all, axis=0)
                else:
                    uv_cat = np.zeros((0, 2), dtype=np.float32)
                print(
                    f"[LLM-TRAJ] sampled_uv_candidates={int(uv_cat.shape[0])} "
                    f"(elapsed={(time.perf_counter()-t0):.3f}s)"
                )

                pts = backproject_uvs_to_3d_points(uv_cat, depth, K)
                print(
                    f"[LLM-TRAJ] backproject_valid_pts={int(pts.shape[0])} "
                    f"(elapsed={(time.perf_counter()-t0):.3f}s)"
                )
                # 去重（粗略）并取前 5 个
                if pts.shape[0] > 0:
                    # 基于四舍五入到 1mm 做唯一化
                    key = np.round(pts / 0.001).astype(np.int64)
                    _, uniq_idx = np.unique(key, axis=0, return_index=True)
                    pts = pts[np.sort(uniq_idx)]
                pick_points_cam = pts[:5].astype(np.float32)
                print(
                    f"[LLM-TRAJ] pick_points_cam_n={int(pick_points_cam.shape[0])} "
                    f"(elapsed={(time.perf_counter()-t0):.3f}s)"
                )

                if pick_points_cam.shape[0] < 5:
                    raise RuntimeError(
                        f"mask 回投有效 3D 点不足 5 个（当前 {pick_points_cam.shape[0]} 个）。"
                    )

                traj_planner = self.traj_planner or DeepSeekR1TrajectoryPlanner()
                print(
                    f"[LLM-TRAJ] 调用 DeepSeek-R1 规划中... "
                    f"(model={getattr(traj_planner, 'model', 'unknown')})"
                )
                t_call = time.perf_counter()
                llm_traj = traj_planner.plan_trajectory(
                    task_pick=pick_text,
                    task_place=place_text,
                    pick_points_cam=pick_points_cam,
                    place_point_cam=place_point_cam,
                    T_cam_grasp=T_cam_grasp,
                    T_cam_place=T_cam_place,
                )
                dt_call = time.perf_counter() - t_call
                if isinstance(llm_traj, dict):
                    wps = llm_traj.get("waypoints")
                    n_wps = len(wps) if isinstance(wps, list) else 0
                    print(
                        f"[LLM-TRAJ] DeepSeek 返回成功: keys={list(llm_traj.keys())}, "
                        f"waypoints={n_wps}, call_elapsed={dt_call:.3f}s, "
                        f"total_elapsed={(time.perf_counter()-t0):.3f}s"
                    )
                else:
                    print(
                        f"[LLM-TRAJ] DeepSeek 返回非 dict: type={type(llm_traj)}, "
                        f"call_elapsed={dt_call:.3f}s, total_elapsed={(time.perf_counter()-t0):.3f}s"
                    )
            except Exception as e:
                # 轨迹规划失败不应影响抓取/放置位姿输出
                print(f"[LLM-TRAJ] 轨迹规划失败: {e}")
                llm_traj = {"error": str(e)}

        result: Dict[str, Any] = {
            "grasp_pose_cam": T_cam_grasp,
            "place_pose_cam": T_cam_place,
            "pick_mask": pick_mask,
            "cloud_o3d": cloud_o3d,
            "grasp_group": gg_best,
            "pick_bbox": [x1, y1, x2, y2],
            "pick_center_uv": pick_center_uv,
            "place_bbox": [px1, py1, px2, py2],
            "place_center_uv": (place_u, place_v),
            "place_point_cam": place_point_cam,
            "pick_points_cam": pick_points_cam,
            "llm_traj": llm_traj,
            "vlm_pick_response": pick_res.get("response"),
            "vlm_place_response": place_res.get("response"),
        }

        if return_motion:
            if T_cam_init is None:
                raise ValueError("return_motion=True 时必须提供 T_cam_init")
            steps = build_default_motion_steps_for_pick_place(
                T_cam_grasp=T_cam_grasp,
                T_cam_place=T_cam_place,
                T_cam_init=T_cam_init,
                grip_open=grip_open,
            )
            # 若 LLM 返回了轨迹，用它替换默认的“移动到放置位姿”步骤（保留抓取闭合/放开/回初始交互）
            if return_llm_traj and isinstance(llm_traj, dict) and "error" not in llm_traj:
                llm_steps = build_motion_steps_from_llm_traj(
                    llm_traj,
                    T_cam_grasp=T_cam_grasp,
                    default_grip_after_grasp=0.0,
                )
                if len(llm_steps) > 0 and len(steps) >= 5:
                    # 默认 steps 结构：
                    # 0 move grasp(open) / 1 close / 2 move place(keep) / 3 open / 4 return init
                    # 用 llm_steps 替换 index=2，并将“放开夹爪”位置改为 llm 轨迹最后一点
                    steps = [steps[0], steps[1]] + llm_steps + [steps[3], steps[4]]
                    steps[-2].T_cam = llm_steps[-1].T_cam  # 放开夹爪跟随最后轨迹点
            result["motion_steps"] = steps

        return result


@dataclass
class MotionStep:
    """
    表示单个“相机系下的移动 + 夹爪”动作。
    """

    T_cam: np.ndarray
    grip: float
    desc: str
    apply_graspnet_corr: bool = True
    prompt: Optional[str] = None
    keep_grip: bool = False
    require_prev_executed: bool = False


def build_default_motion_steps_for_pick_place(
    *,
    T_cam_grasp: np.ndarray,
    T_cam_place: np.ndarray,
    T_cam_init: np.ndarray,
    grip_open: float,
) -> List[MotionStep]:
    """
    根据抓取/放置/初始位姿，构造一个默认的交互式 pick&place 动作序列。

    说明：
    - 本函数只负责“生成步骤”，不负责与机械臂通信/执行。
    - 生成的步骤仍带有 prompt，方便上层（apps）做交互式确认。
    - 之后若要“直接输出轨迹”，可把每个 MotionStep 进一步细化为多个 waypoints。
    """
    return [
        # Step 1: 始终先移动到抓取位姿（张开夹爪）
        MotionStep(
            T_cam=T_cam_grasp,
            grip=grip_open,
            desc="移动到抓取位姿(张开夹爪)",
        ),
        # Step 2: 是否在当前位置闭合夹爪
        MotionStep(
            T_cam=T_cam_grasp,
            grip=0.0,
            desc="闭合夹爪(保持当前位置和姿态不变)",
            prompt=(
                "\n[online] 是否执行抓取（在当前位置闭合夹爪）？"
                "按回车执行抓取；输入 n 后回车则跳过闭合夹爪: "
            ),
        ),
        # Step 3: 是否移动到放置位姿（保持当前夹爪状态）
        MotionStep(
            T_cam=T_cam_place,
            grip=0.0,  # 实际发送时保持 current_grip
            desc="移动到放置位姿(保持当前夹爪开合状态)",
            prompt=(
                "\n[online] 是否移动到放置位置？"
                "按回车移动到放置位姿；输入 n 后回车则跳过本次放置: "
            ),
            keep_grip=True,
        ),
        # Step 4: 只有在已经移动到放置位姿后，才询问是否放开夹爪
        MotionStep(
            T_cam=T_cam_place,
            grip=grip_open,
            desc="放开夹爪(保持当前位置和姿态不变)",
            prompt=(
                "\n[online] 是否在当前放置位姿处放开夹爪？"
                "按回车放开夹爪；输入 n 后回车则保持当前夹爪状态: "
            ),
            require_prev_executed=True,
        ),
        # Step 5: 无论是否移动到放置位姿，最后都询问一次是否回到预设初始位姿
        MotionStep(
            T_cam=T_cam_init,
            grip=0.0,  # 实际发送时保持 current_grip
            desc="回到预设初始位姿",
            prompt=(
                "\n[online] 是否回到预设初始位姿？"
                "按回车回到初始位姿；输入 n 后回车则停留在当前位置: "
            ),
            keep_grip=True,
            apply_graspnet_corr=False,
        ),
    ]


def zeroshot_plan(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    pick_text: str,
    place_text: str,
) -> Dict[str, Any]:
    """
    为了兼容之前的函数式调用方式，保留一个薄封装，
    内部会创建一个默认配置的 ZeroShotPlanner 并调用其 plan()。
    """
    planner = ZeroShotPlanner()
    return planner.plan(
        color_bgr=color_bgr,
        depth=depth,
        K=K,
        pick_text=pick_text,
        place_text=place_text,
    )


__all__ = [
    "ZeroShotPlanner",
    "GraspEstimator",
    "VLMClient",
    "SAMSegmenter",
    "DeepSeekR1TrajectoryPlanner",
    "zeroshot_plan",
    "backproject_pixel_to_3d",
    "sample_mask_pixels",
    "backproject_uvs_to_3d_points",
    "build_motion_steps_from_llm_traj",
    "MotionStep",
    "build_default_motion_steps_for_pick_place",
]

