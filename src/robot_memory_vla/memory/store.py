import json
from dataclasses import asdict
from pathlib import Path

from robot_memory_vla.runtime.models import MemoryItem


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def read_all(self) -> list[MemoryItem]:
        items: list[MemoryItem] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            items.append(MemoryItem(**json.loads(line)))
        return items

    def append_item(self, item: MemoryItem) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
