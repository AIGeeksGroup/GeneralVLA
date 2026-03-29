from datetime import datetime
from pathlib import Path

import robot_memory_vla.runtime.logger as logger_module
from robot_memory_vla.runtime.logger import RunLogger
from robot_memory_vla.runtime.models import MemoryItem
from robot_memory_vla.runtime.task_interpreter import TaskInterpreter


def test_task_interpreter_extracts_pick_and_place_targets() -> None:
    interpreter = TaskInterpreter()
    result = interpreter.interpret(
        "抓起桌面上的瓶盖，放到右下角粉色盒子上",
        [
            MemoryItem(
                task_id="task-1",
                task_text="抓起桌面上的瓶盖，放到盒子上",
                outcome="success",
                failure_reason=None,
                memory_text="瓶盖优先抓上沿。",
            )
        ],
    )
    assert result.pick_target_text == "桌面上的瓶盖"
    assert result.place_target_text == "右下角粉色盒子上"
    assert "瓶盖优先抓上沿" in result.success_hint


def test_run_logger_creates_run_directory(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path)
    artifacts = logger.start_run("抓起桌面上的瓶盖，放到右下角粉色盒子上")
    assert artifacts.run_dir.exists()
    assert artifacts.run_dir.parent.name.count("-") == 2


def test_run_logger_creates_unique_directories_with_same_timestamp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FrozenDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 3, 29, 12, 0, 0)

    monkeypatch.setattr(logger_module, "datetime", FrozenDateTime)
    logger = RunLogger(tmp_path)

    first = logger.start_run("任务一")
    second = logger.start_run("任务二")

    assert first.run_dir != second.run_dir
