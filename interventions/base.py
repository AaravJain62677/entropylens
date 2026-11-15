from dataclasses import dataclass
from enum import Enum


class InterventionType(str, Enum):
    CONTROL = "control"
    NONE = "none"
    STRONG_EVIDENCE = "strong_evidence"
    WEAK_EVIDENCE = "weak_evidence"
    SOCIAL_PRESSURE = "social_pressure"
    MISLEADING_EVIDENCE = "misleading_evidence"
    AMBIGUITY = "ambiguity"


@dataclass
class Intervention:
    type: InterventionType
    content: str = ""

    def apply(self, prompt: str) -> str:
        """
        Apply an intervention to an existing prompt.

        Control/none conditions leave the prompt unchanged.
        """

        if self.type in {
            InterventionType.CONTROL,
            InterventionType.NONE,
        }:
            return prompt

        return (
            f"{prompt}\n\n"
            f"{self.content}"
        )