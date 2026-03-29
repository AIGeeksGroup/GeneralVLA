from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

import numpy as np

from robot_memory_vla.memory.store import MemoryStore
from robot_memory_vla.runtime.models import MemoryItem, TaskMemoryRecord


def _simple_embed_text(text: str, dims: int = 16) -> np.ndarray:
    vector = np.zeros(dims, dtype=float)
    for char in text:
        vector[ord(char) % dims] += 1.0
    return vector


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


class ReasoningBankAdapter:
    def __init__(self, store: MemoryStore, embed_text: Callable[[str], np.ndarray]) -> None:
        self.store = store
        self.embed_text = embed_text

    @classmethod
    def from_reasoning_bank(
        cls,
        store: MemoryStore,
        reasoning_bank_root: str,
        backend: str,
    ) -> "ReasoningBankAdapter":
        if backend.lower() == "simple":
            return cls(store=store, embed_text=_simple_embed_text)

        module_path = Path(reasoning_bank_root) / "WebArena" / "memory_management.py"
        spec = importlib.util.spec_from_file_location(
            "reasoning_bank_memory_management",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load ReasoningBank memory module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if backend.lower() == "qwen":
            embed_text = lambda text: module.embed_query_with_qwen(text).squeeze(0).numpy()
        else:
            embed_text = lambda text: module.embed_query_with_gemini(text).squeeze(0).numpy()

        return cls(store=store, embed_text=embed_text)

    def retrieve(self, task_text: str, top_k: int = 3) -> list[MemoryItem]:
        query = self.embed_text(task_text)
        scored: list[tuple[float, MemoryItem]] = []
        for item in self.store.read_all():
            if item.embedding is None:
                continue
            score = _cosine_similarity(query, np.asarray(item.embedding, dtype=float))
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def write(self, record: TaskMemoryRecord) -> None:
        embedding = self.embed_text(record.task_text).tolist()
        self.store.append_item(
            MemoryItem(
                task_id=record.task_id,
                task_text=record.task_text,
                outcome=record.outcome,
                failure_reason=record.failure_reason,
                memory_text=record.memory_text,
                tags=[],
                embedding=embedding,
            )
        )
