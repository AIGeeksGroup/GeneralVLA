from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RobotConfig:
    host: str
    port: int
    grip_open: float
    init_xyz_mm: list[float]
    init_rpy_deg: list[float]


@dataclass
class ModelConfig:
    reasoning_bank_root: str
    generalvla_root: str
    zeroshotpick_root: str
    generalvla_vis_save_path: str
    retrieval_backend: str
    generalvla_lisa_version: str = ""
    generalvla_segagent_version: str = ""
    generalvla_simpleclick_checkpoint: str = ""
    generalvla_grounding_model: str = "qwen-full"
    generalvla_seg_model: str = "simple_click"
    generalvla_precision: str = "fp16"
    generalvla_device: str = "cuda:0"
    generalvla_load_in_4bit: bool = True
    zeroshotpick_graspnet_root: str = "/data2/Project/Arm/ycliu/VLM_Grasp_Interactive"
    zeroshotpick_graspnet_checkpoint_path: str = ""


@dataclass
class RuntimeConfig:
    data_root: str
    memory_path: str
    top_k: int
    require_operator_confirmation: bool


@dataclass
class AppConfig:
    robot: RobotConfig
    models: ModelConfig
    runtime: RuntimeConfig


@dataclass
class MemoryItem:
    task_id: str
    task_text: str
    outcome: str
    failure_reason: str | None
    memory_text: str
    tags: list[str] = field(default_factory=list)
    embedding: list[float] | None = None


@dataclass
class TaskMemoryRecord:
    task_id: str
    task_text: str
    outcome: str
    failure_reason: str | None
    memory_text: str
    retrieved_memory_ids: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskInterpretation:
    pick_target_text: str
    pick_part_text: str | None
    place_target_text: str
    success_hint: str


@dataclass
class CaptureFrame:
    color_bgr: np.ndarray
    depth_m: np.ndarray
    k_matrix: np.ndarray


@dataclass
class SegmentationResult:
    mask: np.ndarray
    score: float | None
    text_response: str | None
    debug_images: dict[str, str] = field(default_factory=dict)


@dataclass
class GraspPlan:
    grasp_pose_cam: np.ndarray
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlacePlan:
    place_pose_cam: np.ndarray
    motion_steps: list[Any]
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    success: bool
    failure_reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunArtifacts:
    run_dir: Path
    files: dict[str, Path] = field(default_factory=dict)
