from dataclasses import dataclass
from enum import Enum


class InterventionType(str, Enum):
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
        Apply the intervention to an existing prompt.

        The original question is preserved and the intervention
        is appended to the prompt.
        """

        if self.type == InterventionType.NONE:
            return prompt

        return (
            f"{prompt}\n\n"
            f"{self.content}"
        )