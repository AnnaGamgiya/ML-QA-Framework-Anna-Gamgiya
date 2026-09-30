from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CheckStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class AssertionResult:
    """
    Результат одного объективного assertion внутри QA-check.
    """

    requirement: str
    requirement_source: str
    expected: Any
    actual: Any
    status: CheckStatus
    evidence: Any = None
    reason: str = ""


@dataclass
class ValidationResult:
    """
    Стандартизированный результат одного QA-check.
    """

    check_id: str
    check_name: str
    status: CheckStatus

    requirement: str = ""
    requirement_source: str = ""

    expected: Any = None
    actual: Any = None

    evidence: Any = None
    reason: str = ""

    assertions: list[AssertionResult] = field(
        default_factory=list
    )