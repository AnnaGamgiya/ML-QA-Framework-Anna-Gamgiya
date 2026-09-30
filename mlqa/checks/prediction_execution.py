from typing import Any

from mlqa.check import Check
from mlqa.context import ModelContract
from mlqa.result import (
    AssertionResult,
    CheckStatus,
    ValidationResult,
)


class PredictionExecutionCheck(Check):
    """
    Check 2 — Выполнение предсказания.

    Проверяет только техническую возможность
    выполнить model.predict(X) на переданном X.

    Структура и содержимое результата предсказания здесь
    не валидируются. Это ответственность Check 3.
    """

    @property
    def check_id(self) -> str:
        return "prediction_execution"

    @property
    def check_name(self) -> str:
        return "Выполнение предсказания"

    def run(
        self,
        *,
        contract: ModelContract,
        model: Any,
        X: Any,
        y: Any | None = None,
    ) -> ValidationResult:
        if model is None:
            return self._skip(
                reason="Артефакт модели не предоставлен.",
            )

        if X is None:
            return self._skip(
                reason="Входные данные X не предоставлены.",
            )

        predict_method = getattr(
            model,
            "predict",
            None,
        )

        if not callable(predict_method):
            return self._skip(
                reason=(
                    "Модель не предоставляет доступный "
                    "интерфейс predict()."
                ),
            )

        try:
            predict_method(X)

        except Exception as exc:
            assertion = AssertionResult(
                requirement=(
                    "Готовая модель должна обработать переданные "
                    "входные данные без технической ошибки."
                ),
                requirement_source="формальный API модели",
                expected=(
                    "Обработка тестовых данных завершается "
                    "без исключения"
                ),
                actual=type(exc).__name__,
                status=CheckStatus.FAIL,
                evidence={
                    "этап": "predict",
                    "тип_исключения": type(exc).__name__,
                    "сообщение_исключения": str(exc),
                },
                reason=(
                    "При обработке тестовых данных моделью "
                    "возникло исключение."
                ),
            )

            return self._result(
                status=CheckStatus.FAIL,
                assertion=assertion,
            )

        assertion = AssertionResult(
            requirement=(
                "Готовая модель должна обработать переданные "
                "входные данные без технической ошибки."
            ),
            requirement_source="формальный API модели",
            expected=(
                "Обработка тестовых данных завершается "
                "без исключения"
            ),
            actual=(
                "Тестовые данные обработаны моделью "
                "без исключения"
            ),
            status=CheckStatus.PASS,
            evidence={
                "этап": "predict",
                "исключение": None,
            },
            reason=(
                "Модель успешно обработала тестовые данные "
                "без технической ошибки."
            ),
        )

        return self._result(
            status=CheckStatus.PASS,
            assertion=assertion,
        )

    def _skip(
        self,
        *,
        reason: str,
    ) -> ValidationResult:
        assertion = AssertionResult(
            requirement=(
                "Готовая модель должна обработать переданные "
                "входные данные без технической ошибки."
            ),
            requirement_source="формальный API модели",
            expected=(
                "Доступна возможность получить предсказания "
                "для тестовых данных"
            ),
            actual=None,
            status=CheckStatus.SKIP,
            evidence=None,
            reason=reason,
        )

        return self._result(
            status=CheckStatus.SKIP,
            assertion=assertion,
        )

    def _result(
        self,
        *,
        status: CheckStatus,
        assertion: AssertionResult,
    ) -> ValidationResult:
        return ValidationResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            requirement=assertion.requirement,
            requirement_source=assertion.requirement_source,
            expected=assertion.expected,
            actual=assertion.actual,
            evidence=assertion.evidence,
            reason=assertion.reason,
            assertions=[assertion],
        )