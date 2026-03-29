import argparse
from pathlib import Path

from robot_memory_vla.adapters.generalvla_adapter import GeneralVLAAdapter
from robot_memory_vla.adapters.reasoning_bank_adapter import ReasoningBankAdapter
from robot_memory_vla.adapters.zeroshotpick_adapter import ZeroShotPickAdapter
from robot_memory_vla.app.config import default_config_dir, load_app_config, validate_app_config
from robot_memory_vla.app.orchestrator import RobotMemoryVLAOrchestrator
from robot_memory_vla.memory.store import MemoryStore
from robot_memory_vla.runtime.logger import RunLogger
from robot_memory_vla.runtime.task_interpreter import TaskInterpreter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robot Memory VLA")
    parser.add_argument("--task", help="Single-turn Chinese task text")
    parser.add_argument(
        "--config-dir",
        default=str(default_config_dir()),
        help="Directory containing robot.yaml, models.yaml, runtime.yaml",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate configured paths and Python dependencies without running the robot flow",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_app_config(Path(args.config_dir))
    if args.preflight:
        issues = validate_app_config(config)
        if issues:
            for issue in issues:
                print(issue)
            return 1
        print("Preflight OK")
        return 0
    if not args.task:
        parser.error("--task is required unless --preflight is used")
    store = MemoryStore(Path(config.runtime.memory_path))
    memory_adapter = ReasoningBankAdapter.from_reasoning_bank(
        store=store,
        reasoning_bank_root=config.models.reasoning_bank_root,
        backend=config.models.retrieval_backend,
    )
    generalvla_adapter = GeneralVLAAdapter(
        generalvla_root=config.models.generalvla_root,
        vis_save_path=Path(config.models.generalvla_vis_save_path),
        lisa_version=config.models.generalvla_lisa_version,
        segagent_version=config.models.generalvla_segagent_version,
        simpleclick_checkpoint=config.models.generalvla_simpleclick_checkpoint,
        grounding_model=config.models.generalvla_grounding_model,
        seg_model=config.models.generalvla_seg_model,
        precision=config.models.generalvla_precision,
        device=config.models.generalvla_device,
        load_in_4bit=config.models.generalvla_load_in_4bit,
    )
    zeroshotpick_adapter = ZeroShotPickAdapter(
        zeroshotpick_root=config.models.zeroshotpick_root,
        host=config.robot.host,
        port=config.robot.port,
        grip_open=config.robot.grip_open,
        init_xyz_mm=config.robot.init_xyz_mm,
        init_rpy_deg=config.robot.init_rpy_deg,
        graspnet_root=config.models.zeroshotpick_graspnet_root,
        graspnet_checkpoint_path=config.models.zeroshotpick_graspnet_checkpoint_path,
    )
    orchestrator = RobotMemoryVLAOrchestrator(
        memory_adapter=memory_adapter,
        generalvla_adapter=generalvla_adapter,
        zeroshotpick_adapter=zeroshotpick_adapter,
        interpreter=TaskInterpreter(),
        run_logger=RunLogger(Path(config.runtime.data_root)),
        top_k=config.runtime.top_k,
        require_operator_confirmation=config.runtime.require_operator_confirmation,
    )
    result = orchestrator.run(args.task)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
