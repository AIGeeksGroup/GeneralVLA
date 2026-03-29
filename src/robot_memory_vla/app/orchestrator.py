from __future__ import annotations

from copy import copy
from dataclasses import asdict
from uuid import uuid4

from robot_memory_vla.runtime.logger import RunLogger
from robot_memory_vla.runtime.models import ExecutionResult, TaskMemoryRecord


class RobotMemoryVLAOrchestrator:
    def __init__(
        self,
        memory_adapter,
        generalvla_adapter,
        zeroshotpick_adapter,
        interpreter,
        run_logger: RunLogger,
        top_k: int,
        require_operator_confirmation: bool,
    ) -> None:
        self.memory_adapter = memory_adapter
        self.generalvla_adapter = generalvla_adapter
        self.zeroshotpick_adapter = zeroshotpick_adapter
        self.interpreter = interpreter
        self.run_logger = run_logger
        self.top_k = top_k
        self.require_operator_confirmation = require_operator_confirmation

    def _apply_execution_policy(self, plan):
        if self.require_operator_confirmation:
            return plan

        sanitized_steps = []
        for step in plan.motion_steps:
            if hasattr(step, "prompt"):
                step = copy(step)
                step.prompt = None
            sanitized_steps.append(step)
        plan.motion_steps = sanitized_steps
        return plan

    def run(self, task_text: str) -> ExecutionResult:
        run = self.run_logger.start_run(task_text)
        frame = self.zeroshotpick_adapter.capture()
        self.run_logger.write_color(run, "capture_rgb.jpg", frame.color_bgr)

        memories = self.memory_adapter.retrieve(task_text, top_k=self.top_k)
        self.run_logger.write_json(
            run,
            "retrieved_memories.json",
            {"items": [asdict(item) for item in memories]},
        )

        interpretation = self.interpreter.interpret(task_text, memories)
        object_seg = self.generalvla_adapter.segment_pick_object(
            frame.color_bgr,
            task_text,
            interpretation.pick_target_text,
        )
        self.run_logger.write_mask(run, "object_mask.png", object_seg.mask)

        grasp_seg = self.generalvla_adapter.segment_grasp_region(
            frame.color_bgr,
            object_seg.mask,
            interpretation.pick_part_text,
        )
        self.run_logger.write_mask(run, "grasp_mask.png", grasp_seg.mask)

        grasp_plan = self.zeroshotpick_adapter.plan_grasp(
            frame,
            grasp_seg.mask,
            interpretation.pick_target_text,
        )
        place_plan = self.zeroshotpick_adapter.plan_place(
            frame,
            interpretation.place_target_text,
            grasp_plan,
        )
        place_plan = self._apply_execution_policy(place_plan)
        execution = self.zeroshotpick_adapter.execute(place_plan)

        memory_record = TaskMemoryRecord(
            task_id=str(uuid4()),
            task_text=task_text,
            outcome="success" if execution.success else "failure",
            failure_reason=execution.failure_reason,
            memory_text=interpretation.success_hint or task_text,
            retrieved_memory_ids=[item.task_id for item in memories],
            extra={"run_dir": str(run.run_dir)},
        )
        self.memory_adapter.write(memory_record)
        self.run_logger.write_json(
            run,
            "execution_result.json",
            {
                "success": execution.success,
                "failure_reason": execution.failure_reason,
                "metadata": execution.metadata,
            },
        )
        return execution
