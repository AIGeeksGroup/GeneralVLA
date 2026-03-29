from pathlib import Path
from typing import Optional

import numpy as np

from robot_memory_vla.app.orchestrator import RobotMemoryVLAOrchestrator
from robot_memory_vla.runtime.logger import RunLogger
from robot_memory_vla.runtime.models import (
    CaptureFrame,
    ExecutionResult,
    GraspPlan,
    MemoryItem,
    PlacePlan,
    SegmentationResult,
    TaskInterpretation,
)


class FakeStep:
    def __init__(self, prompt: Optional[str]) -> None:
        self.prompt = prompt


class FakeMemoryAdapter:
    def __init__(self) -> None:
        self.written = []

    def retrieve(self, task_text: str, top_k: int = 3):
        return [
            MemoryItem(
                task_id="task-1",
                task_text="抓起瓶盖，放到盒子上",
                outcome="success",
                failure_reason=None,
                memory_text="瓶盖优先抓上沿。",
            )
        ]

    def write(self, record):
        self.written.append(record)


class FakeInterpreter:
    def interpret(self, task_text: str, memories):
        return TaskInterpretation(
            pick_target_text="桌面上的瓶盖",
            pick_part_text="瓶盖上沿",
            place_target_text="右下角粉色盒子上",
            success_hint="瓶盖优先抓上沿。",
        )


class FakeGeneralVLA:
    def segment_pick_object(self, image_bgr, task_text, pick_target_text):
        return SegmentationResult(mask=np.ones((2, 2), dtype=np.uint8), score=None, text_response="obj")

    def segment_grasp_region(self, image_bgr, object_mask, pick_part_text):
        return SegmentationResult(mask=np.ones((2, 2), dtype=np.uint8), score=None, text_response="grasp")


class FakeZeroShotPick:
    def __init__(self) -> None:
        self.executed_plan = None

    def capture(self):
        return CaptureFrame(
            color_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
            depth_m=np.ones((2, 2), dtype=np.float32),
            k_matrix=np.eye(3, dtype=np.float32),
        )

    def plan_grasp(self, frame, grasp_mask, pick_text):
        return GraspPlan(grasp_pose_cam=np.eye(4, dtype=np.float32))

    def plan_place(self, frame, place_text, grasp_plan):
        return PlacePlan(
            place_pose_cam=np.eye(4, dtype=np.float32),
            motion_steps=[FakeStep(prompt="confirm"), FakeStep(prompt=None)],
        )

    def execute(self, plan):
        self.executed_plan = plan
        return ExecutionResult(success=True, failure_reason=None, metadata={})


def test_orchestrator_runs_full_pipeline_and_writes_memory(tmp_path: Path) -> None:
    memory = FakeMemoryAdapter()
    zeroshotpick = FakeZeroShotPick()
    orchestrator = RobotMemoryVLAOrchestrator(
        memory_adapter=memory,
        generalvla_adapter=FakeGeneralVLA(),
        zeroshotpick_adapter=zeroshotpick,
        interpreter=FakeInterpreter(),
        run_logger=RunLogger(tmp_path),
        top_k=3,
        require_operator_confirmation=False,
    )

    result = orchestrator.run("抓起桌面上的瓶盖，放到右下角粉色盒子上")

    assert result.success is True
    assert len(memory.written) == 1
    assert [step.prompt for step in zeroshotpick.executed_plan.motion_steps] == [None, None]
