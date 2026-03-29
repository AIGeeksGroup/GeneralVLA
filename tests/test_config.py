from pathlib import Path

from robot_memory_vla.app.config import (
    _default_generalvla_lisa_version,
    default_config_dir,
    repository_root,
    load_app_config,
    validate_app_config,
)


def test_load_app_config_reads_all_three_files(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "robot.yaml").write_text(
        "host: 10.5.23.176\n"
        "port: 8888\n"
        "grip_open: 50.0\n"
        "init_xyz_mm: [-28.12, -200.0, 371.47]\n"
        "init_rpy_deg: [0.0, 0.0, -98.87]\n",
        encoding="utf-8",
    )
    (config_dir / "models.yaml").write_text(
        "reasoning_bank_root: vendor/reasoning-bank\n"
        "generalvla_root: vendor/GeneralVLA\n"
        "zeroshotpick_root: vendor/zeroshotpick-main\n"
        "generalvla_vis_save_path: data/vis\n"
        "retrieval_backend: stub\n",
        encoding="utf-8",
    )
    (config_dir / "runtime.yaml").write_text(
        "data_root: data\n"
        "memory_path: data/memories.jsonl\n"
        "top_k: 3\n"
        "require_operator_confirmation: false\n",
        encoding="utf-8",
    )

    config = load_app_config(config_dir)

    assert config.robot.host == "10.5.23.176"
    assert Path(config.models.generalvla_root) == tmp_path / "vendor" / "GeneralVLA"
    assert Path(config.models.reasoning_bank_root) == tmp_path / "vendor" / "reasoning-bank"
    assert Path(config.runtime.data_root) == tmp_path / "data"
    assert config.runtime.top_k == 3
    assert config.robot.init_rpy_deg == [0.0, 0.0, -98.87]


def test_validate_app_config_reports_missing_paths_and_modules(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "robot.yaml").write_text(
        "host: 10.5.23.176\n"
        "port: 8888\n"
        "grip_open: 50.0\n"
        "init_xyz_mm: [-28.12, -200.0, 371.47]\n"
        "init_rpy_deg: [0.0, 0.0, -98.87]\n",
        encoding="utf-8",
    )
    (config_dir / "models.yaml").write_text(
        "reasoning_bank_root: /missing/reasoning-bank\n"
        "generalvla_root: /missing/GeneralVLA\n"
        "zeroshotpick_root: /missing/zeroshotpick-main\n"
        "generalvla_vis_save_path: /tmp/vis\n"
        "retrieval_backend: gemini\n",
        encoding="utf-8",
    )
    (config_dir / "runtime.yaml").write_text(
        "data_root: /tmp/data\n"
        "memory_path: /tmp/memories.jsonl\n"
        "top_k: 3\n"
        "require_operator_confirmation: false\n",
        encoding="utf-8",
    )

    config = load_app_config(config_dir)
    issues = validate_app_config(
        config,
        module_checker=lambda name: False,
        path_checker=lambda path: False,
    )

    assert any("reasoning_bank_root" in issue for issue in issues)
    assert any("generalvla_root" in issue for issue in issues)
    assert any("zeroshotpick_root" in issue for issue in issues)
    assert any("google.genai" in issue for issue in issues)
    assert any("vertexai" in issue for issue in issues)
    assert any("torch" in issue for issue in issues)
    assert any("tensorboard" in issue for issue in issues)
    assert any("easydict" in issue for issue in issues)
    assert any("albumentations" in issue for issue in issues)


def test_validate_app_config_reports_missing_runtime_model_assets(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    reasoning_bank_root = tmp_path / "reasoning-bank"
    generalvla_root = tmp_path / "GeneralVLA"
    zeroshotpick_root = tmp_path / "zeroshotpick-main"
    reasoning_bank_root.mkdir()
    generalvla_root.mkdir()
    zeroshotpick_root.mkdir()
    (config_dir / "robot.yaml").write_text(
        "host: 10.5.23.176\n"
        "port: 8888\n"
        "grip_open: 50.0\n"
        "init_xyz_mm: [-28.12, -200.0, 371.47]\n"
        "init_rpy_deg: [0.0, 0.0, -98.87]\n",
        encoding="utf-8",
    )
    (config_dir / "models.yaml").write_text(
        f"reasoning_bank_root: {reasoning_bank_root}\n"
        f"generalvla_root: {generalvla_root}\n"
        f"zeroshotpick_root: {zeroshotpick_root}\n"
        "generalvla_vis_save_path: /tmp/vis\n"
        "retrieval_backend: simple\n",
        encoding="utf-8",
    )
    (config_dir / "runtime.yaml").write_text(
        "data_root: /tmp/data\n"
        "memory_path: /tmp/memories.jsonl\n"
        "top_k: 3\n"
        "require_operator_confirmation: false\n",
        encoding="utf-8",
    )

    config = load_app_config(config_dir)
    issues = validate_app_config(config, module_checker=lambda name: True)

    assert any("generalvla_lisa_version" in issue for issue in issues)
    assert any("generalvla_segagent_version" in issue for issue in issues)
    assert any("generalvla_simpleclick_checkpoint" in issue for issue in issues)
    assert any("zeroshotpick_graspnet_root" in issue for issue in issues)
    assert any("zeroshotpick_graspnet_checkpoint_path" in issue for issue in issues)


def test_shipped_models_config_uses_lisa_7b_explanatory() -> None:
    config = load_app_config(default_config_dir())

    assert config.models.generalvla_lisa_version.endswith("/LISA-7B-v1-explanatory")


def test_default_generalvla_lisa_version_uses_7b_explanatory() -> None:
    config = load_app_config(default_config_dir())
    config.models.generalvla_lisa_version = ""

    assert str(_default_generalvla_lisa_version(config)).endswith("/LISA-7B-v1-explanatory")


def test_shipped_config_does_not_use_machine_specific_paths() -> None:
    config_dir = default_config_dir()
    for name in ("models.yaml", "runtime.yaml"):
        text = (config_dir / name).read_text(encoding="utf-8")
        assert "/home/why/reasoning-bank" not in text
        assert "/home/why/GeneralVLA" not in text
        assert "/home/why/zeroshotpick-main" not in text
        assert "/home/why/robot-memory-vla" not in text


def test_repository_root_matches_current_repo() -> None:
    root = repository_root()

    assert root.name == "GeneralVLA v1 simplified"
    assert (root / "configs").exists()
