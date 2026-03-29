from pathlib import Path

import numpy as np

from robot_memory_vla.adapters.reasoning_bank_adapter import ReasoningBankAdapter
from robot_memory_vla.memory.store import MemoryStore
from robot_memory_vla.runtime.models import MemoryItem, TaskMemoryRecord


def test_reasoning_bank_adapter_returns_most_similar_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "robot_tasks.jsonl")
    store.append_item(
        MemoryItem(
            task_id="task-1",
            task_text="抓起桌面上的瓶盖，放到粉色盒子上",
            outcome="success",
            failure_reason=None,
            memory_text="瓶盖体积小，优先抓顶部边缘。",
            tags=["bottle-cap"],
            embedding=[1.0, 0.0],
        )
    )
    store.append_item(
        MemoryItem(
            task_id="task-2",
            task_text="抓起桌面上的杯子，放到篮子里",
            outcome="success",
            failure_reason=None,
            memory_text="杯口易滑，抓杯身中部。",
            tags=["cup"],
            embedding=[0.0, 1.0],
        )
    )

    adapter = ReasoningBankAdapter(
        store=store,
        embed_text=lambda text: np.array([1.0, 0.0], dtype=float),
    )

    result = adapter.retrieve("抓起桌面上的瓶盖，放到右下角盒子上", top_k=1)

    assert [item.task_id for item in result] == ["task-1"]


def test_reasoning_bank_adapter_writes_task_record(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "robot_tasks.jsonl")
    adapter = ReasoningBankAdapter(
        store=store,
        embed_text=lambda text: np.array([0.5, 0.5], dtype=float),
    )

    adapter.write(
        TaskMemoryRecord(
            task_id="task-3",
            task_text="抓起瓶盖，放到盒子上",
            outcome="success",
            failure_reason=None,
            memory_text="盒子边缘较高，放置点要偏中心。",
        )
    )

    loaded = store.read_all()
    assert len(loaded) == 1
    assert loaded[0].task_id == "task-3"


def test_reasoning_bank_adapter_supports_simple_backend_without_reasoning_bank_files(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "robot_tasks.jsonl")
    adapter = ReasoningBankAdapter.from_reasoning_bank(
        store=store,
        reasoning_bank_root=str(tmp_path / "missing-root"),
        backend="simple",
    )

    adapter.write(
        TaskMemoryRecord(
            task_id="task-4",
            task_text="抓起桌面上的瓶盖，放到盒子上",
            outcome="success",
            failure_reason=None,
            memory_text="优先抓上沿。",
        )
    )

    result = adapter.retrieve("抓起瓶盖，放到右下角盒子上", top_k=1)
    assert len(result) == 1
    assert result[0].task_id == "task-4"
