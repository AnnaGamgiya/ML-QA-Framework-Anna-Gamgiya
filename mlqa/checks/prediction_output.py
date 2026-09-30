from typing import Any

import numpy as np

from mlqa.check import Check
from mlqa.context import ModelContract
from mlqa.result import (
    AssertionResult,
    CheckStatus,
    ValidationResult,
)


PROBABILITY_SUM_TOLERANCE = 1e-7


class PredictionOutputCheck(Check):
    """
    Check 3 — Результат предсказания.

    Проверяет структуру и формальные свойства
    результата работы модели.
    """

    @property
    def check_id(self) -> str:
        return "prediction_output"

    @property
    def check_name(self) -> str:
        return "Результат предсказания"

    def run(
        self,
        *,
        contract: ModelContract,
        model: Any,
        X: Any,
        y: Any | None = None,
    ) -> ValidationResult:
        if model is None:
            return self._skip_result(
                "Артефакт модели не предоставлен."
            )

        if X is None:
            return self._skip_result(
                "Входные данные X не предоставлены."
            )

        predict_method = getattr(model, "predict", None)

        if not callable(predict_method):
            return self._skip_result(
                "Модель не предоставляет доступный интерфейс predict()."
            )

        try:
            predictions = predict_method(X)
        except Exception:
            return self._skip_result(
                "Предсказания не были успешно получены; "
                "Check 3 зависит от успешного результата Check 2."
            )

        assertions: list[AssertionResult] = []

        assertions.append(
            self._check_prediction_row_count(
                predictions=predictions,
                X=X,
            )
        )

        assertions.append(
            self._check_prediction_classes(
                predictions=predictions,
                contract=contract,
            )
        )

        predict_proba_method = getattr(
            model,
            "predict_proba",
            None,
        )

        if callable(predict_proba_method):
            try:
                probabilities = predict_proba_method(X)
            except Exception as exc:
                assertions.append(
                    AssertionResult(
                        requirement=(
                            "Вероятности должны быть доступны, "
                            "если модель предоставляет predict_proba()."
                        ),
                        requirement_source="формальный API модели",
                        expected=(
                            "Получение вероятностей завершается "
                            "без исключения"
                        ),
                        actual=type(exc).__name__,
                        status=CheckStatus.FAIL,
                        evidence={
                            "этап": "predict_proba",
                            "тип_исключения": type(exc).__name__,
                            "сообщение_исключения": str(exc),
                        },
                        reason=(
                            "При получении вероятностей "
                            "возникло исключение."
                        ),
                    )
                )
            else:
                assertions.extend(
                    self._check_probabilities(
                        probabilities=probabilities,
                        X=X,
                        contract=contract,
                    )
                )

        return self._build_result(assertions)

    @staticmethod
    def _check_prediction_row_count(
        *,
        predictions: Any,
        X: Any,
    ) -> AssertionResult:
        expected = len(X)

        try:
            actual = len(predictions)
        except TypeError:
            actual = None

        if actual != expected:
            return AssertionResult(
                requirement=(
                    "Количество предсказаний должно совпадать "
                    "с количеством строк во входных данных."
                ),
                requirement_source="контракт результата framework v1",
                expected=expected,
                actual=actual,
                status=CheckStatus.FAIL,
                evidence={
                    "ожидаемое_количество_строк": expected,
                    "количество_предсказаний": actual,
                },
                reason=(
                    "Количество предсказаний не совпадает "
                    "с количеством строк во входных данных."
                ),
            )

        return AssertionResult(
            requirement=(
                "Количество предсказаний должно совпадать "
                "с количеством строк во входных данных."
            ),
            requirement_source="контракт результата framework v1",
            expected=expected,
            actual=actual,
            status=CheckStatus.PASS,
            evidence={
                "ожидаемое_количество_строк": expected,
                "количество_предсказаний": actual,
            },
            reason=(
                "Количество предсказаний совпадает "
                "с количеством строк во входных данных."
            ),
        )

    @staticmethod
    def _check_prediction_classes(
        *,
        predictions: Any,
        contract: ModelContract,
    ) -> AssertionResult:
        classes_field = contract.classes

        if classes_field is None:
            return AssertionResult(
                requirement=(
                    "Предсказанные классы должны принадлежать "
                    "множеству классов модели."
                ),
                requirement_source="формальный API модели",
                expected=None,
                actual=None,
                status=CheckStatus.SKIP,
                reason=(
                    "Формальная информация о классах модели "
                    "недоступна."
                ),
            )

        expected_classes = list(classes_field.value)

        try:
            predicted_values = np.asarray(predictions).reshape(-1)
            invalid_values = [
                value.item()
                if hasattr(value, "item")
                else value
                for value in predicted_values
                if value not in expected_classes
            ]
        except Exception as exc:
            return AssertionResult(
                requirement=(
                    "Предсказанные классы должны принадлежать "
                    "множеству классов модели."
                ),
                requirement_source=classes_field.source,
                expected=expected_classes,
                actual=None,
                status=CheckStatus.FAIL,
                evidence={
                    "тип_исключения": type(exc).__name__,
                    "сообщение_исключения": str(exc),
                },
                reason=(
                    "Не удалось проверить значения "
                    "предсказанных классов."
                ),
            )

        if invalid_values:
            return AssertionResult(
                requirement=(
                    "Предсказанные классы должны принадлежать "
                    "множеству классов модели."
                ),
                requirement_source=classes_field.source,
                expected=expected_classes,
                actual=invalid_values,
                status=CheckStatus.FAIL,
                evidence={
                    "недопустимые_предсказания": invalid_values,
                },
                reason=(
                    "Результат предсказания содержит значения, "
                    "которые отсутствуют среди классов модели."
                ),
            )

        return AssertionResult(
            requirement=(
                "Предсказанные классы должны принадлежать "
                "множеству классов модели."
            ),
            requirement_source=classes_field.source,
            expected=expected_classes,
            actual=(
                "Все предсказания принадлежат "
                "множеству классов модели"
            ),
            status=CheckStatus.PASS,
            evidence={
                "недопустимые_предсказания": [],
            },
            reason=(
                "Все предсказанные классы принадлежат "
                "множеству классов модели."
            ),
        )

    def _check_probabilities(
        self,
        *,
        probabilities: Any,
        X: Any,
        contract: ModelContract,
    ) -> list[AssertionResult]:
        assertions: list[AssertionResult] = []

        try:
            probability_array = np.asarray(
                probabilities,
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            return [
                AssertionResult(
                    requirement=(
                        "Результат predict_proba() должен содержать "
                        "числовые значения вероятностей."
                    ),
                    requirement_source="формальный API модели",
                    expected="Числовая матрица вероятностей",
                    actual=type(exc).__name__,
                    status=CheckStatus.FAIL,
                    evidence={
                        "тип_исключения": type(exc).__name__,
                        "сообщение_исключения": str(exc),
                    },
                    reason=(
                        "Результат predict_proba() невозможно "
                        "интерпретировать как числовые значения."
                    ),
                )
            ]

        actual_rows = (
            probability_array.shape[0]
            if probability_array.ndim >= 1
            else None
        )

        expected_rows = len(X)

        assertions.append(
            AssertionResult(
                requirement=(
                    "Количество строк с вероятностями должно "
                    "совпадать с количеством строк во входных данных."
                ),
                requirement_source="контракт результата framework v1",
                expected=expected_rows,
                actual=actual_rows,
                status=(
                    CheckStatus.PASS
                    if actual_rows == expected_rows
                    else CheckStatus.FAIL
                ),
                evidence={
                    "ожидаемое_количество_строк": expected_rows,
                    "количество_строк_вероятностей": actual_rows,
                },
                reason=(
                    "Количество строк с вероятностями совпадает "
                    "с количеством строк во входных данных."
                    if actual_rows == expected_rows
                    else
                    "Количество строк с вероятностями не совпадает "
                    "с количеством строк во входных данных."
                ),
            )
        )

        classes_field = contract.classes

        if classes_field is None:
            assertions.append(
                AssertionResult(
                    requirement=(
                        "Количество столбцов с вероятностями должно "
                        "совпадать с количеством классов модели."
                    ),
                    requirement_source="формальный API модели",
                    expected=None,
                    actual=(
                        probability_array.shape[1]
                        if probability_array.ndim == 2
                        else None
                    ),
                    status=CheckStatus.SKIP,
                    reason=(
                        "Формальная информация о классах модели "
                        "недоступна."
                    ),
                )
            )
        else:
            expected_columns = len(classes_field.value)
            actual_columns = (
                probability_array.shape[1]
                if probability_array.ndim == 2
                else None
            )

            assertions.append(
                AssertionResult(
                    requirement=(
                        "Количество столбцов с вероятностями должно "
                        "совпадать с количеством классов модели."
                    ),
                    requirement_source=classes_field.source,
                    expected=expected_columns,
                    actual=actual_columns,
                    status=(
                        CheckStatus.PASS
                        if actual_columns == expected_columns
                        else CheckStatus.FAIL
                    ),
                    evidence={
                        "ожидаемое_количество_столбцов": expected_columns,
                        "количество_столбцов_вероятностей": actual_columns,
                    },
                    reason=(
                        "Количество столбцов с вероятностями совпадает "
                        "с количеством классов модели."
                        if actual_columns == expected_columns
                        else
                        "Количество столбцов с вероятностями не совпадает "
                        "с количеством классов модели."
                    ),
                )
            )

        finite_mask = np.isfinite(probability_array)
        all_finite = bool(np.all(finite_mask))

        non_finite_indices = [
            list(index)
            for index in np.argwhere(~finite_mask)
        ]

        assertions.append(
            AssertionResult(
                requirement=(
                    "Все значения вероятностей должны быть конечными."
                ),
                requirement_source="контракт результата framework v1",
                expected="Все значения конечные",
                actual=all_finite,
                status=(
                    CheckStatus.PASS
                    if all_finite
                    else CheckStatus.FAIL
                ),
                evidence={
                    "индексы_неконечных_значений": non_finite_indices,
                },
                reason=(
                    "Все значения вероятностей являются конечными."
                    if all_finite
                    else
                    "Результат содержит неконечные "
                    "значения вероятностей."
                ),
            )
        )

        in_range_mask = (
            (probability_array >= 0.0)
            & (probability_array <= 1.0)
        )
        all_in_range = bool(np.all(in_range_mask))

        out_of_range_indices = [
            list(index)
            for index in np.argwhere(~in_range_mask)
        ]

        assertions.append(
            AssertionResult(
                requirement=(
                    "Каждая вероятность должна находиться "
                    "в диапазоне [0, 1]."
                ),
                requirement_source="контракт результата framework v1",
                expected="[0, 1]",
                actual=all_in_range,
                status=(
                    CheckStatus.PASS
                    if all_in_range
                    else CheckStatus.FAIL
                ),
                evidence={
                    "индексы_значений_вне_диапазона": (
                        out_of_range_indices
                    ),
                },
                reason=(
                    "Все вероятности находятся в диапазоне [0, 1]."
                    if all_in_range
                    else
                    "Результат содержит вероятности "
                    "вне диапазона [0, 1]."
                ),
            )
        )

        if probability_array.ndim != 2:
            assertions.append(
                AssertionResult(
                    requirement=(
                        "Сумма вероятностей в каждой строке должна "
                        "быть равна 1 с учётом фиксированного "
                        "численного допуска."
                    ),
                    requirement_source="контракт результата framework v1",
                    expected={
                        "сумма_строки": 1.0,
                        "абсолютный_допуск": (
                            PROBABILITY_SUM_TOLERANCE
                        ),
                    },
                    actual={
                        "размерность": probability_array.ndim,
                    },
                    status=CheckStatus.FAIL,
                    evidence={
                        "недопустимая_размерность": (
                            probability_array.ndim
                        ),
                    },
                    reason=(
                        "Результат predict_proba() не является "
                        "двумерной матрицей."
                    ),
                )
            )

            return assertions

        row_sums = probability_array.sum(axis=1)

        valid_row_sums = np.isclose(
            row_sums,
            1.0,
            rtol=0.0,
            atol=PROBABILITY_SUM_TOLERANCE,
        )

        invalid_rows = np.flatnonzero(
            ~valid_row_sums
        ).tolist()

        assertions.append(
            AssertionResult(
                requirement=(
                    "Сумма вероятностей в каждой строке должна "
                    "быть равна 1 с учётом фиксированного "
                    "численного допуска."
                ),
                requirement_source="контракт результата framework v1",
                expected={
                    "сумма_строки": 1.0,
                    "абсолютный_допуск": (
                        PROBABILITY_SUM_TOLERANCE
                    ),
                },
                actual={
                    "строки_с_нарушением": invalid_rows,
                },
                status=(
                    CheckStatus.PASS
                    if not invalid_rows
                    else CheckStatus.FAIL
                ),
                evidence={
                    "проверено_строк": len(row_sums),
                    "количество_строк_с_нарушением": len(invalid_rows),
                    "строки_с_нарушением": invalid_rows[:10],
                    "абсолютный_допуск": (
                        PROBABILITY_SUM_TOLERANCE
                    ),
                },
                reason=(
                    "Сумма вероятностей во всех строках равна 1 "
                    "с учётом фиксированного численного допуска."
                    if not invalid_rows
                    else
                    "Хотя бы в одной строке сумма вероятностей "
                    "не равна 1 с учётом фиксированного "
                    "численного допуска."
                ),
            )
        )

        return assertions

    def _skip_result(
        self,
        reason: str,
    ) -> ValidationResult:
        assertion = AssertionResult(
            requirement=(
                "Результат предсказания должен соответствовать "
                "формальному контракту результата."
            ),
            requirement_source="контракт результата framework v1",
            expected="Формальный контракт результата соблюдён",
            actual=None,
            status=CheckStatus.SKIP,
            reason=reason,
        )

        return ValidationResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=CheckStatus.SKIP,
            requirement=assertion.requirement,
            requirement_source=assertion.requirement_source,
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
            if assertion.status is not CheckStatus.SKIP
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
                "Результат предсказания должен соответствовать "
                "формальному контракту результата."
            ),
            requirement_source=(
                "формальный API модели и контракт "
                "результата framework v1"
            ),
            expected="Формальный контракт результата соблюдён",
            actual=status.value,
            evidence={
                "количество_assertions": len(assertions),
                "выполнено_assertions": len(executed),
            },
            reason=self._result_reason(status),
            assertions=assertions,
        )

    @staticmethod
    def _result_reason(
        status: CheckStatus,
    ) -> str:
        reasons = {
            CheckStatus.PASS: (
                "Все применимые проверки результата "
                "предсказания пройдены."
            ),
            CheckStatus.WARN: (
                "Хотя бы одна применимая проверка результата "
                "предсказания завершилась предупреждением."
            ),
            CheckStatus.FAIL: (
                "Хотя бы одна применимая проверка результата "
                "предсказания не пройдена."
            ),
            CheckStatus.SKIP: (
                "Ни одна проверка результата предсказания "
                "не могла быть выполнена."
            ),
        }

        return reasons[status]