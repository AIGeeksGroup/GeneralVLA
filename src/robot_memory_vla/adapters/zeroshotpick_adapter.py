from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from robot_memory_vla.runtime.models import CaptureFrame, ExecutionResult, GraspPlan, PlacePlan


class ZeroShotPickAdapter:
    def __init__(
        self,
        zeroshotpick_root: str,
        host: str,
        port: int,
        grip_open: float,
        init_xyz_mm: list[float],
        init_rpy_deg: list[float],
        graspnet_root: str = "/data2/Project/Arm/ycliu/VLM_Grasp_Interactive",
        graspnet_checkpoint_path: str = "",
        client_factory: Callable[[str, int], object] | None = None,
        grasp_estimator_factory: Callable[[], object] | None = None,
        vlm_client_factory: Callable[[], object] | None = None,
        se3_from_grasp: Callable[[object], np.ndarray] | None = None,
        backproject_pixel_to_3d: Callable[[float, float, np.ndarray, np.ndarray], np.ndarray] | None = None,
        build_motion_steps: Callable[..., list] | None = None,
        execute_motion_sequence: Callable[[object, list, float], float] | None = None,
    ) -> None:
        self.zeroshotpick_root = Path(zeroshotpick_root)
        self.host = host
        self.port = port
        self.grip_open = grip_open
        self.init_xyz_mm = init_xyz_mm
        self.init_rpy_deg = init_rpy_deg
        self.graspnet_root = Path(graspnet_root)
        self.graspnet_checkpoint_path = Path(graspnet_checkpoint_path) if graspnet_checkpoint_path else None
        self._client_factory = client_factory
        self._grasp_estimator_factory = grasp_estimator_factory
        self._vlm_client_factory = vlm_client_factory
        self._se3_from_grasp = se3_from_grasp
        self._backproject_pixel_to_3d = backproject_pixel_to_3d
        self._build_motion_steps = build_motion_steps
        self._execute_motion_sequence = execute_motion_sequence
        self._client = None

    @staticmethod
    def _ensure_numpy_compat() -> None:
        legacy_aliases = {
            "float": float,
            "int": int,
            "bool": bool,
            "complex": complex,
            "object": object,
        }
        for name, value in legacy_aliases.items():
            if name not in np.__dict__:
                setattr(np, name, value)

    def _resolved_graspnet_checkpoint_path(self) -> Path:
        if self.graspnet_checkpoint_path is not None:
            return self.graspnet_checkpoint_path
        return self.graspnet_root / "logs" / "log_rs" / "checkpoint-rs.tar"

    def _inject_graspnet_paths(self) -> None:
        baseline_root = self.graspnet_root / "graspnet-baseline"
        for path in (
            baseline_root / "models",
            baseline_root / "dataset",
            baseline_root / "utils",
        ):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

    def _load_module(self, filename: str, module_name: str):
        module_path = self.zeroshotpick_root / filename
        self._ensure_numpy_compat()
        self._inject_graspnet_paths()
        if str(self.zeroshotpick_root) not in sys.path:
            sys.path.insert(0, str(self.zeroshotpick_root))
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load zeroshot module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _get_client(self):
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory(self.host, self.port)
            else:
                client_module = self._load_module("upper_client.py", "zeroshotpick_upper_client")
                self._client = client_module.UpperClient(self.host, self.port)
        return self._client

    def _get_grasp_estimator(self):
        if self._grasp_estimator_factory is not None:
            return self._grasp_estimator_factory()

        pipeline = self._load_module("zeroshot_pipeline.py", "zeroshotpick_pipeline")
        if hasattr(pipeline, "ROOT_VLM_GRASP"):
            pipeline.ROOT_VLM_GRASP = str(self.graspnet_root)
        if hasattr(pipeline, "GRASPNET_LOG_CKPT"):
            pipeline.GRASPNET_LOG_CKPT = str(self._resolved_graspnet_checkpoint_path())
        if self._se3_from_grasp is None:
            self._se3_from_grasp = pipeline.se3_from_grasp_in_cam
        if self._backproject_pixel_to_3d is None:
            self._backproject_pixel_to_3d = pipeline.backproject_pixel_to_3d
        if self._build_motion_steps is None:
            self._build_motion_steps = pipeline.build_default_motion_steps_for_pick_place
        if self._vlm_client_factory is None:
            self._vlm_client_factory = pipeline.VLMClient
        return pipeline.GraspEstimator(checkpoint_path=str(self._resolved_graspnet_checkpoint_path()))

    @staticmethod
    def _euler_zyx_deg_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
        rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
        cr, sr = np.cos(rx), np.sin(rx)
        cp, sp = np.cos(ry), np.sin(ry)
        cy, sy = np.cos(rz), np.sin(rz)
        rz_mat = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        ry_mat = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
        rx_mat = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
        return rz_mat @ ry_mat @ rx_mat

    def _build_init_pose(self) -> np.ndarray:
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = self._euler_zyx_deg_to_matrix(*self.init_rpy_deg)
        pose[:3, 3] = np.asarray(self.init_xyz_mm, dtype=np.float32) / 1000.0
        return pose

    def capture(self) -> CaptureFrame:
        color_bgr, depth_m, k_matrix = self._get_client().capture()
        return CaptureFrame(color_bgr=color_bgr, depth_m=depth_m, k_matrix=k_matrix)

    def plan_grasp(self, frame: CaptureFrame, grasp_mask: np.ndarray, pick_text: str) -> GraspPlan:
        estimator = self._get_grasp_estimator()
        best_grasp, _ = estimator.estimate_best_grasp_with_cloud(
            frame.color_bgr,
            frame.depth_m,
            frame.k_matrix,
            grasp_mask,
        )
        if self._se3_from_grasp is None:
            raise RuntimeError("se3_from_grasp function is not available")
        pose = self._se3_from_grasp(best_grasp)
        return GraspPlan(grasp_pose_cam=pose, debug={"pick_text": pick_text})

    def plan_place(self, frame: CaptureFrame, place_text: str, grasp_plan: GraspPlan) -> PlacePlan:
        if self._vlm_client_factory is None or self._backproject_pixel_to_3d is None or self._build_motion_steps is None:
            self._get_grasp_estimator()
        if self._vlm_client_factory is None or self._backproject_pixel_to_3d is None or self._build_motion_steps is None:
            raise RuntimeError("Place-planning dependencies are not available")

        vlm_client = self._vlm_client_factory()
        place_result = vlm_client.locate_bbox(place_text, frame.color_bgr)
        bbox = place_result["coordinates"]["bbox"]
        center_u = (bbox[0] + bbox[2]) / 2.0
        center_v = (bbox[1] + bbox[3]) / 2.0
        place_point = self._backproject_pixel_to_3d(center_u, center_v, frame.depth_m, frame.k_matrix)

        place_pose = np.eye(4, dtype=np.float32)
        place_pose[:3, :3] = grasp_plan.grasp_pose_cam[:3, :3]
        place_pose[:3, 3] = place_point

        motion_steps = self._build_motion_steps(
            T_cam_grasp=grasp_plan.grasp_pose_cam,
            T_cam_place=place_pose,
            T_cam_init=self._build_init_pose(),
            grip_open=self.grip_open,
        )
        return PlacePlan(
            place_pose_cam=place_pose,
            motion_steps=motion_steps,
            debug={"place_text": place_text, "bbox": bbox},
        )

    def execute(self, plan: PlacePlan) -> ExecutionResult:
        if self._execute_motion_sequence is None:
            apps_module = self._load_module("zeroshot_apps.py", "zeroshotpick_apps")
            self._execute_motion_sequence = apps_module.execute_motion_sequence

        try:
            self._execute_motion_sequence(self._get_client(), plan.motion_steps, self.grip_open)
        except Exception as exc:
            return ExecutionResult(success=False, failure_reason=str(exc), metadata={})

        return ExecutionResult(success=True, failure_reason=None, metadata={})
