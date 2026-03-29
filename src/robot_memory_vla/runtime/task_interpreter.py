import re

from robot_memory_vla.runtime.models import MemoryItem, TaskInterpretation


class TaskInterpreter:
    _pattern = re.compile(r"(?:抓起|拿起|夹起|拾起|把)(.*?)(?:放到|放在|放入|放进|放去|置于)(.*)")

    def interpret(self, task_text: str, memories: list[MemoryItem]) -> TaskInterpretation:
        text = task_text.strip().rstrip("。.")
        match = self._pattern.search(text)
        if match:
            pick_target = match.group(1).strip(" ，,")
            place_target = match.group(2).strip(" ，,")
        else:
            pick_target = text
            place_target = text
        success_hint = " ".join(item.memory_text for item in memories)
        return TaskInterpretation(
            pick_target_text=pick_target,
            pick_part_text=None,
            place_target_text=place_target,
            success_hint=success_hint,
        )
