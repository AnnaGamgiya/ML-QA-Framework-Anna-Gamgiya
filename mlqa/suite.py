from typing import Any

from mlqa.check import Check
from mlqa.context import ModelContract
from mlqa.result import ValidationResult


class QASuite:
    """
    Последовательно запускает QA-checks
    над одним набором model / X / y / contract.

    Каждый check самостоятельно определяет
    применимость assertions и итоговый статус.
    """

    def __init__(self, checks: list[Check]):
        self.checks = list(checks)

    def run(
        self,
        *,
        contract: ModelContract,
        model: Any,
        X: Any,
        y: Any | None = None,
    ) -> list[ValidationResult]:
        """
        Запускает checks в заданном порядке
        и возвращает их ValidationResult.
        """

        results: list[ValidationResult] = []

        for check in self.checks:
            result = check.run(
                contract=contract,
                model=model,
                X=X,
                y=y,
            )
            results.append(result)

        return results