import json
from datetime import datetime
from pathlib import Path

import numpy as np

from robot_memory_vla.runtime.models import RunArtifacts


class RunLogger:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root

    def start_run(self, task_text: str) -> RunArtifacts:
        del task_text
        now = datetime.now()
        day_dir = self.data_root / "runs" / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        run_name = now.strftime("%H%M%S")
        run_dir = day_dir / run_name
        suffix = 1
        while run_dir.exists():
            run_dir = day_dir / f"{run_name}_{suffix:02d}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        return RunArtifacts(run_dir=run_dir, files={})

    def write_json(self, run: RunArtifacts, name: str, payload: dict) -> Path:
        path = run.run_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        run.files[name] = path
        return path

    def write_mask(self, run: RunArtifacts, name: str, mask: np.ndarray) -> Path:
        import cv2

        path = run.run_dir / name
        cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
        run.files[name] = path
        return path

    def write_color(self, run: RunArtifacts, name: str, color_bgr: np.ndarray) -> Path:
        import cv2

        path = run.run_dir / name
        cv2.imwrite(str(path), color_bgr)
        run.files[name] = path
        return path
