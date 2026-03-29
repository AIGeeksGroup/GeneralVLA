import numpy as np
from pathlib import Path

from robot_memory_vla.adapters.zeroshotpick_adapter import ZeroShotPickAdapter
from robot_memory_vla.runtime.models import CaptureFrame, GraspPlan


class FakeClient:
    def capture(self):
        color = np.zeros((2, 2, 3), dtype=np.uint8)
        depth = np.ones((2, 2), dtype=np.float32)
        k_matrix = np.eye(3, dtype=np.float32)
        return color, depth, k_matrix


class FakeGraspEstimator:
    def estimate_best_grasp_with_cloud(self, color_bgr, depth, k_matrix, mask):
        return object(), object()


def fake_se3_from_grasp(_):
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = 0.2
    return pose


class FakeVLMClient:
    def locate_bbox(self, text, color_bgr):
        return {"coordinates": {"bbox": [0, 0, 1, 1]}, "response": text}


def fake_backproject(u, v, depth, k_matrix):
    return np.array([0.1, 0.2, 0.3], dtype=np.float32)


def fake_build_steps(**kwargs):
    return ["move-grasp", "close", "move-place", "open", "return"]


def test_zeroshotpick_adapter_wraps_capture_and_planning() -> None:
    adapter = ZeroShotPickAdapter(
        zeroshotpick_root="/home/why/zeroshotpick-main",
        host="10.5.23.176",
        port=8888,
        grip_open=50.0,
        init_xyz_mm=[-28.12, -200.0, 371.47],
        init_rpy_deg=[0.0, 0.0, -98.87],
        client_factory=lambda host, port: FakeClient(),
        grasp_estimator_factory=lambda: FakeGraspEstimator(),
        vlm_client_factory=lambda: FakeVLMClient(),
        se3_from_grasp=fake_se3_from_grasp,
        backproject_pixel_to_3d=fake_backproject,
        build_motion_steps=fake_build_steps,
        execute_motion_sequence=lambda client, steps, initial_grip: initial_grip,
    )

    frame = adapter.capture()
    grasp_plan = adapter.plan_grasp(frame, np.ones((2, 2), dtype=np.uint8), "瓶盖")
    place_plan = adapter.plan_place(frame, "右下角粉色盒子上", grasp_plan)

    assert isinstance(frame, CaptureFrame)
    assert isinstance(grasp_plan, GraspPlan)
    assert place_plan.motion_steps == ["move-grasp", "close", "move-place", "open", "return"]


def test_zeroshotpick_adapter_loads_pipeline_with_external_graspnet_paths(tmp_path: Path) -> None:
    zeroshot_root = tmp_path / "zeroshot"
    zeroshot_root.mkdir()
    external_root = tmp_path / "external"
    graspnet_models = external_root / "graspnet-baseline" / "models"
    graspnet_dataset = external_root / "graspnet-baseline" / "dataset"
    graspnet_utils = external_root / "graspnet-baseline" / "utils"
    graspnet_models.mkdir(parents=True)
    graspnet_dataset.mkdir(parents=True)
    graspnet_utils.mkdir(parents=True)
    (graspnet_models / "graspnet.py").write_text(
        "class GraspNet:\n"
        "    pass\n"
        "def pred_decode(*args, **kwargs):\n"
        "    return 'decoded'\n",
        encoding="utf-8",
    )
    (zeroshot_root / "zeroshot_pipeline.py").write_text(
        "from graspnet import GraspNet, pred_decode\n"
        "class GraspEstimator:\n"
        "    def __init__(self, checkpoint_path=''):\n"
        "        self.checkpoint_path = checkpoint_path\n"
        "def se3_from_grasp_in_cam(grasp):\n"
        "    return grasp\n"
        "def backproject_pixel_to_3d(u, v, depth, k_matrix):\n"
        "    return depth[0, 0]\n"
        "def build_default_motion_steps_for_pick_place(**kwargs):\n"
        "    return ['move']\n"
        "class VLMClient:\n"
        "    pass\n",
        encoding="utf-8",
    )
    checkpoint_path = external_root / "logs" / "log_rs" / "checkpoint-rs.tar"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text("stub", encoding="utf-8")

    adapter = ZeroShotPickAdapter(
        zeroshotpick_root=str(zeroshot_root),
        host="10.5.23.176",
        port=8888,
        grip_open=50.0,
        init_xyz_mm=[-28.12, -200.0, 371.47],
        init_rpy_deg=[0.0, 0.0, -98.87],
        graspnet_root=str(external_root),
        graspnet_checkpoint_path=str(checkpoint_path),
    )

    estimator = adapter._get_grasp_estimator()

    assert estimator.checkpoint_path == str(checkpoint_path)
    assert adapter._build_motion_steps is not None
    assert adapter._se3_from_grasp is not None


def test_zeroshotpick_adapter_adds_numpy_legacy_aliases() -> None:
    if "float" in np.__dict__:
        delattr(np, "float")
    if "int" in np.__dict__:
        delattr(np, "int")
    if "bool" in np.__dict__:
        delattr(np, "bool")

    ZeroShotPickAdapter._ensure_numpy_compat()

    assert np.float is float
    assert np.int is int
    assert np.bool is bool
