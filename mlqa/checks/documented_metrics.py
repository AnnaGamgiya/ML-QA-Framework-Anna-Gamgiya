from decimal import Decimal
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from mlqa.check import Check
from mlqa.context import ModelContract
from mlqa.result import (
    AssertionResult,
    CheckStatus,
    ValidationResult,
)


MetricFunction = Callable[[np.ndarray, np.ndarray], float]


METRIC_FUNCTIONS: dict[str, MetricFunction] = {
    "ROC-AUC": roc_auc_score,
    "PR-AUC": average_precision_score,
    "Brier": brier_score_loss,
}


class DocumentedMetricsCheck(Check):
    """
    Check 4 — воспроизводимость документированных метрик.

    Проверяет, воспроизводятся ли документированные метрики
    для переданных модели, X и y.

    Конкретные значения проекта или датасета
    в CORE не зашиваются.
    """

    @property
    def check_id(self) -> str:
        return "documented_metrics"

    @property
    def check_name(self) -> str:
        return "Воспроизводимость документированных метрик"

    def run(
        self,
        *,
        contract: ModelContract,
        model: Any,
        X: Any,
        y: Any | None = None,
    ) -> ValidationResult:
        if y is None:
            return self._skip_result(
                "Целевые значения y не предоставлены."
            )

        if model is None:
            return self._skip_result(
                "Артефакт модели не предоставлен."
            )

        if X is None:
            return self._skip_result(
                "Тестовые данные X не предоставлены."
            )

        metrics_field = contract.shipped_metrics

        if metrics_field is None:
            return self._skip_result(
                "В структурированных metadata "
                "не указаны документированные метрики."
            )

        expected_metrics = dict(metrics_field.value)

        if not expected_metrics:
            return self._skip_result(
                "В структурированных metadata "
                "нет документированных метрик."
            )

        applicable_metric_names = [
            metric_name
            for metric_name in expected_metrics
            if metric_name in METRIC_FUNCTIONS
        ]

        assertions: list[AssertionResult] = []

        for metric_name, expected_value in expected_metrics.items():
            if metric_name not in METRIC_FUNCTIONS:
                assertions.append(
                    self._unsupported_metric_assertion(
                        metric_name=metric_name,
                        expected_value=expected_value,
                        source=metrics_field.source,
                    )
                )

        if not applicable_metric_names:
            return self._build_result(assertions)

        predict_proba_method = getattr(
            model,
            "predict_proba",
            None,
        )

        if not callable(predict_proba_method):
            for metric_name in applicable_metric_names:
                assertions.append(
                    self._metric_skip_assertion(
                        metric_name=metric_name,
                        expected_value=expected_metrics[metric_name],
                        source=metrics_field.source,
                        reason=(
                            "Модель не предоставляет вероятности, "
                            "необходимые для воспроизведения "
                            "этой метрики."
                        ),
                    )
                )

            return self._build_result(assertions)

        classes_field = contract.classes

        if classes_field is None:
            for metric_name in applicable_metric_names:
                assertions.append(
                    self._metric_skip_assertion(
                        metric_name=metric_name,
                        expected_value=expected_metrics[metric_name],
                        source=metrics_field.source,
                        reason=(
                            "Формальная информация о классах "
                            "модели недоступна."
                        ),
                    )
                )

            return self._build_result(assertions)

        classes = list(classes_field.value)

        if len(classes) != 2:
            for metric_name in applicable_metric_names:
                assertions.append(
                    self._metric_skip_assertion(
                        metric_name=metric_name,
                        expected_value=expected_metrics[metric_name],
                        source=metrics_field.source,
                        reason=(
                            "Для воспроизведения этой метрики "
                            "в Check 4 v1 требуется бинарная "
                            "классификация."
                        ),
                    )
                )

            return self._build_result(assertions)

        positive_class = classes[1]
        positive_class_index = classes.index(
            positive_class
        )

        try:
            probabilities = np.asarray(
                predict_proba_method(X),
                dtype=float,
            )
        except Exception:
            for metric_name in applicable_metric_names:
                assertions.append(
                    self._metric_skip_assertion(
                        metric_name=metric_name,
                        expected_value=expected_metrics[metric_name],
                        source=metrics_field.source,
                        reason=(
                            "Не удалось получить вероятности модели. "
                            "Воспроизведение метрики зависит "
                            "от успешного получения результата модели."
                        ),
                    )
                )

            return self._build_result(assertions)

        if (
            probabilities.ndim != 2
            or probabilities.shape[1] != len(classes)
        ):
            for metric_name in applicable_metric_names:
                assertions.append(
                    self._metric_skip_assertion(
                        metric_name=metric_name,
                        expected_value=expected_metrics[metric_name],
                        source=metrics_field.source,
                        reason=(
                            "Структура вероятностей не соответствует "
                            "формальной информации о классах модели."
                        ),
                    )
                )

            return self._build_result(assertions)

        positive_scores = probabilities[
            :,
            positive_class_index,
        ]

        y_values = self._normalize_y(y)

        if y_values is None:
            for metric_name in applicable_metric_names:
                assertions.append(
                    self._metric_skip_assertion(
                        metric_name=metric_name,
                        expected_value=expected_metrics[metric_name],
                        source=metrics_field.source,
                        reason=(
                            "Целевые значения y невозможно "
                            "однозначно интерпретировать "
                            "как один целевой столбец."
                        ),
                    )
                )

            return self._build_result(assertions)

        if len(y_values) != len(positive_scores):
            for metric_name in applicable_metric_names:
                assertions.append(
                    self._metric_skip_assertion(
                        metric_name=metric_name,
                        expected_value=expected_metrics[metric_name],
                        source=metrics_field.source,
                        reason=(
                            "Количество строк в y не совпадает "
                            "с количеством результатов модели."
                        ),
                    )
                )

            return self._build_result(assertions)

        binary_y = (
            np.asarray(y_values) == positive_class
        ).astype(int)

        for metric_name in applicable_metric_names:
            assertions.append(
                self._evaluate_metric(
                    metric_name=metric_name,
                    expected_value=expected_metrics[metric_name],
                    binary_y=binary_y,
                    positive_scores=positive_scores,
                    positive_class=positive_class,
                    source=metrics_field.source,
                )
            )

        return self._build_result(assertions)

    @staticmethod
    def _normalize_y(
        y: Any,
    ) -> np.ndarray | None:
        if hasattr(y, "to_numpy"):
            values = y.to_numpy()
        else:
            values = np.asarray(y)

        values = np.asarray(values)

        if values.ndim == 1:
            return values

        if (
            values.ndim == 2
            and values.shape[1] == 1
        ):
            return values[:, 0]

        return None

    @staticmethod
    def _unsupported_metric_assertion(
        *,
        metric_name: str,
        expected_value: Any,
        source: str,
    ) -> AssertionResult:
        return AssertionResult(
            requirement=(
                f"Воспроизвести документированную "
                f"метрику {metric_name}."
            ),
            requirement_source=source,
            expected=expected_value,
            actual=None,
            status=CheckStatus.SKIP,
            reason=(
                f"Метрика {metric_name} документирована, "
                "но в Check 4 v1 для неё нет "
                "автоматизированного правила проверки."
            ),
        )

    @staticmethod
    def _metric_skip_assertion(
        *,
        metric_name: str,
        expected_value: Any,
        source: str,
        reason: str,
    ) -> AssertionResult:
        return AssertionResult(
            requirement=(
                f"Воспроизвести документированную "
                f"метрику {metric_name}."
            ),
            requirement_source=source,
            expected=expected_value,
            actual=None,
            status=CheckStatus.SKIP,
            reason=reason,
        )

    def _evaluate_metric(
        self,
        *,
        metric_name: str,
        expected_value: Any,
        binary_y: np.ndarray,
        positive_scores: np.ndarray,
        positive_class: Any,
        source: str,
    ) -> AssertionResult:
        metric_function = METRIC_FUNCTIONS[
            metric_name
        ]

        try:
            actual_value = float(
                metric_function(
                    binary_y,
                    positive_scores,
                )
            )
        except Exception as exc:
            return AssertionResult(
                requirement=(
                    f"Воспроизвести документированную "
                    f"метрику {metric_name}."
                ),
                requirement_source=source,
                expected=expected_value,
                actual=None,
                status=CheckStatus.FAIL,
                evidence={
                    "metric": metric_name,
                    "тип_ошибки": type(exc).__name__,
                    "сообщение_ошибки": str(exc),
                },
                reason=(
                    f"Метрику {metric_name} не удалось "
                    "воспроизвести на предоставленных "
                    "тестовых данных."
                ),
            )

        precision = self._decimal_places(
            expected_value
        )

        rounded_actual = round(
            actual_value,
            precision,
        )

        expected_float = float(
            expected_value
        )

        passed = (
            rounded_actual == expected_float
        )

        return AssertionResult(
            requirement=(
                f"Воспроизвести документированную "
                f"метрику {metric_name}."
            ),
            requirement_source=source,
            expected=expected_float,
            actual=actual_value,
            status=(
                CheckStatus.PASS
                if passed
                else CheckStatus.FAIL
            ),
            evidence={
                "метрика": metric_name,
                "ожидалось": expected_float,
                "получено": actual_value,
                "получено_после_округления": rounded_actual,
                "точность_сравнения": precision,
                "разница": (
                    actual_value - expected_float
                ),
                "положительный_класс": (
                    positive_class.item()
                    if hasattr(
                        positive_class,
                        "item",
                    )
                    else positive_class
                ),
                "результат": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
            },
            reason=(
                f"Метрика {metric_name} воспроизводит "
                "документированное значение."
                if passed
                else
                f"Метрика {metric_name} не воспроизводит "
                "документированное значение."
            ),
        )

    @staticmethod
    def _decimal_places(
        value: Any,
    ) -> int:
        decimal_value = Decimal(
            str(value)
        )

        return max(
            0,
            -decimal_value.as_tuple().exponent,
        )

    def _skip_result(
        self,
        reason: str,
    ) -> ValidationResult:
        assertion = AssertionResult(
            requirement=(
                "Документированные метрики должны "
                "воспроизводиться на предоставленном "
                "тестовом наборе данных."
            ),
            requirement_source="структурированные metadata",
            expected=(
                "Применимые документированные "
                "метрики воспроизведены"
            ),
            actual=None,
            status=CheckStatus.SKIP,
            reason=reason,
        )

        return ValidationResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=CheckStatus.SKIP,
            requirement=assertion.requirement,
            requirement_source=(
                assertion.requirement_source
            ),
            expected=assertion.expected,
            actual=assertion.actual,
            evidence=assertion.evidence,
            reason=assertion.reason,
            assertions=[assertion],
        )

    def _build_result(
        self,
        assertions: list[AssertionResult],
    ) -> ValidationResult:
        executed = [
            assertion
            for assertion in assertions
            if assertion.status
            is not CheckStatus.SKIP
        ]

        if any(
            assertion.status is CheckStatus.FAIL
            for assertion in executed
        ):
            status = CheckStatus.FAIL

        elif any(
            assertion.status is CheckStatus.WARN
            for assertion in executed
        ):
            status = CheckStatus.WARN

        elif executed:
            status = CheckStatus.PASS

        else:
            status = CheckStatus.SKIP

        return ValidationResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            requirement=(
                "Документированные метрики должны "
                "воспроизводиться на предоставленном "
                "тестовом наборе данных."
            ),
            requirement_source="структурированные metadata",
            expected=(
                "Применимые документированные "
                "метрики воспроизведены"
            ),
            actual=status.value,
            evidence={
                "количество_проверок": len(assertions),
                "количество_выполненных_проверок": len(
                    executed
                ),
            },
            reason=self._result_reason(
                status
            ),
            assertions=assertions,
        )

    @staticmethod
    def _result_reason(
        status: CheckStatus,
    ) -> str:
        reasons = {
            CheckStatus.PASS: (
                "Все применимые документированные "
                "метрики воспроизведены."
            ),
            CheckStatus.WARN: (
                "Хотя бы одна применимая метрика "
                "получила предупреждение."
            ),
            CheckStatus.FAIL: (
                "Хотя бы одна применимая документированная "
                "метрика не воспроизведена."
            ),
            CheckStatus.SKIP: (
                "Ни одну документированную метрику "
                "невозможно воспроизвести."
            ),
        }

        return reasons[status]