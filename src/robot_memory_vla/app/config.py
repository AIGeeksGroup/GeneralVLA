from __future__ import annotations

from pathlib import Path
from typing import Callable

import importlib.util

import yaml

from robot_memory_vla.runtime.models import AppConfig, ModelConfig, RobotConfig, RuntimeConfig


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_config_dir() -> Path:
    return repository_root() / "configs"


def _resolve_path_fields(raw: dict, keys: tuple[str, ...], base_dir: Path) -> dict:
    resolved = dict(raw)
    for key in keys:
        value = resolved.get(key)
        if not value:
            continue
        path = Path(value)
        if path.is_absolute():
            continue
        resolved[key] = str((base_dir / path).resolve())
    return resolved


def _default_generalvla_lisa_version(config: AppConfig) -> Path:
    return Path(config.models.generalvla_root) / "pretrain_model" / "LISA-7B-v1-explanatory"


def _default_generalvla_segagent_version(config: AppConfig) -> Path:
    return Path(config.models.generalvla_root) / "pretrain_model" / "segagent" / "SegAgent-Model"


def _default_generalvla_simpleclick_checkpoint(config: AppConfig) -> Path:
    return Path(config.models.generalvla_root) / "pretrain_model" / "simpleclick" / "cocolvis_vit_large.pth"


def _default_zeroshotpick_graspnet_checkpoint_path(config: AppConfig) -> Path:
    return Path(config.models.zeroshotpick_graspnet_root) / "logs" / "log_rs" / "checkpoint-rs.tar"


def load_app_config(config_dir: Path) -> AppConfig:
    config_dir = Path(config_dir).resolve()
    base_dir = config_dir.parent
    robot_raw = _read_yaml(config_dir / "robot.yaml")
    models_raw = _resolve_path_fields(
        _read_yaml(config_dir / "models.yaml"),
        (
            "reasoning_bank_root",
            "generalvla_root",
            "zeroshotpick_root",
            "generalvla_vis_save_path",
            "generalvla_lisa_version",
            "generalvla_segagent_version",
            "generalvla_simpleclick_checkpoint",
            "zeroshotpick_graspnet_root",
            "zeroshotpick_graspnet_checkpoint_path",
        ),
        base_dir,
    )
    runtime_raw = _resolve_path_fields(
        _read_yaml(config_dir / "runtime.yaml"),
        ("data_root", "memory_path"),
        base_dir,
    )
    return AppConfig(
        robot=RobotConfig(**robot_raw),
        models=ModelConfig(**models_raw),
        runtime=RuntimeConfig(**runtime_raw),
    )


def validate_app_config(
    config: AppConfig,
    module_checker: Callable[[str], bool] | None = None,
    path_checker: Callable[[Path], bool] | None = None,
) -> list[str]:
    def has_module(name: str) -> bool:
        if module_checker is not None:
            return module_checker(name)
        return importlib.util.find_spec(name) is not None

    def path_exists(path: Path) -> bool:
        if path_checker is not None:
            return path_checker(path)
        return path.exists()

    issues: list[str] = []
    required_paths = {
        "reasoning_bank_root": Path(config.models.reasoning_bank_root),
        "generalvla_root": Path(config.models.generalvla_root),
        "zeroshotpick_root": Path(config.models.zeroshotpick_root),
    }
    for label, path in required_paths.items():
        if not path_exists(path):
            issues.append(f"Missing required path: {label}={path}")

    retrieval_backend = config.models.retrieval_backend.lower()
    if retrieval_backend == "gemini":
        for module_name in ("google.genai", "vertexai"):
            if not has_module(module_name):
                issues.append(f"Missing required module for gemini retrieval: {module_name}")
    elif retrieval_backend == "qwen":
        for module_name in ("torch", "transformers"):
            if not has_module(module_name):
                issues.append(f"Missing required module for qwen retrieval: {module_name}")

    for module_name in (
        "cv2",
        "numpy",
        "torch",
        "transformers",
        "openai",
        "pycocotools",
        "open3d",
        "ultralytics",
        "tensorboard",
        "easydict",
        "albumentations",
    ):
        if not has_module(module_name):
            issues.append(f"Missing required module for perception/planning: {module_name}")

    generalvla_lisa_version = Path(config.models.generalvla_lisa_version) if config.models.generalvla_lisa_version else _default_generalvla_lisa_version(config)
    generalvla_segagent_version = Path(config.models.generalvla_segagent_version) if config.models.generalvla_segagent_version else _default_generalvla_segagent_version(config)
    generalvla_simpleclick_checkpoint = (
        Path(config.models.generalvla_simpleclick_checkpoint)
        if config.models.generalvla_simpleclick_checkpoint
        else _default_generalvla_simpleclick_checkpoint(config)
    )
    extra_required_paths = {
        "generalvla_demo_path": Path(config.models.generalvla_root) / "demo.py",
        "generalvla_simpleclick_root": Path(config.models.generalvla_root) / "third_party" / "SimpleClick",
        "generalvla_simpleclick_config": Path(config.models.generalvla_root) / "third_party" / "SimpleClick" / "config.yml",
        "generalvla_lisa_version": generalvla_lisa_version,
        "generalvla_segagent_version": generalvla_segagent_version,
        "generalvla_simpleclick_checkpoint": generalvla_simpleclick_checkpoint,
        "zeroshotpick_pipeline_path": Path(config.models.zeroshotpick_root) / "zeroshot_pipeline.py",
        "zeroshotpick_graspnet_root": Path(config.models.zeroshotpick_graspnet_root),
        "zeroshotpick_graspnet_models": Path(config.models.zeroshotpick_graspnet_root) / "graspnet-baseline" / "models",
        "zeroshotpick_graspnet_dataset": Path(config.models.zeroshotpick_graspnet_root) / "graspnet-baseline" / "dataset",
        "zeroshotpick_graspnet_utils": Path(config.models.zeroshotpick_graspnet_root) / "graspnet-baseline" / "utils",
        "zeroshotpick_graspnet_checkpoint_path": (
            Path(config.models.zeroshotpick_graspnet_checkpoint_path)
            if config.models.zeroshotpick_graspnet_checkpoint_path
            else _default_zeroshotpick_graspnet_checkpoint_path(config)
        ),
    }
    for label, path in extra_required_paths.items():
        if not path_exists(path):
            issues.append(f"Missing required path: {label}={path}")

    return issues
