from typing import Any

import pandas as pd

from mlqa.check import Check
from mlqa.context import ModelContract
from mlqa.result import (
    AssertionResult,
    CheckStatus,
    ValidationResult,
)


class InputDataCheck(Check):
    """
    Check 1 — Входные данные.

    Проверяет X только против объективного
    machine-readable input contract.

    Проверка не выводит требования из самого X.
    """

    @property
    def check_id(self) -> str:
        return "input_data"

    @property
    def check_name(self) -> str:
        return "Входные данные"

    def run(
        self,
        *,
        contract: ModelContract,
        model: Any,
        X: Any,
        y: Any | None = None,
    ) -> ValidationResult:
        assertions: list[AssertionResult] = []

        if not isinstance(X, pd.DataFrame):
            assertions.append(
                AssertionResult(
                    requirement="X должен быть pandas DataFrame.",
                    requirement_source="входной интерфейс framework v1",
                    expected="pandas.DataFrame",
                    actual=type(X).__name__,
                    status=CheckStatus.FAIL,
                    evidence={
                        "фактический_тип": type(X).__name__,
                    },
                    reason=(
                        "Переданный X не соответствует "
                        "входному интерфейсу framework."
                    ),
                )
            )

            return self._build_result(assertions)

        assertions.append(
            self._check_required_features(
                contract=contract,
                X=X,
            )
        )

        assertions.append(
            self._check_duplicate_columns(X)
        )

        assertions.append(
            self._check_metadata_vs_model_features(
                contract=contract,
            )
        )

        return self._build_result(assertions)

    @staticmethod
    def _check_required_features(
        *,
        contract: ModelContract,
        X: pd.DataFrame,
    ) -> AssertionResult:
        field = contract.required_features

        if field is None:
            return AssertionResult(
                requirement=(
                    "Все обязательные признаки должны "
                    "присутствовать в X."
                ),
                requirement_source="структурированная metadata",
                expected=None,
                actual=list(X.columns),
                status=CheckStatus.SKIP,
                reason=(
                    "Структурированная metadata не определяет "
                    "обязательные признаки."
                ),
            )

        required = list(field.value)

        missing = [
            feature
            for feature in required
            if feature not in X.columns
        ]

        if missing:
            return AssertionResult(
                requirement=(
                    "Все обязательные признаки должны "
                    "присутствовать в X."
                ),
                requirement_source=field.source,
                expected=required,
                actual=list(X.columns),
                status=CheckStatus.FAIL,
                evidence={
                    "отсутствующие_признаки": missing,
                },
                reason=(
                    "В X отсутствуют обязательные признаки."
                ),
            )

        return AssertionResult(
            requirement=(
                "Все обязательные признаки должны "
                "присутствовать в X."
            ),
            requirement_source=field.source,
            expected=required,
            actual=list(X.columns),
            status=CheckStatus.PASS,
            evidence={
                "отсутствующие_признаки": [],
            },
            reason=(
                "Все обязательные признаки присутствуют в X."
            ),
        )

    @staticmethod
    def _check_duplicate_columns(
        X: pd.DataFrame,
    ) -> AssertionResult:
        duplicated = X.columns[
            X.columns.duplicated(keep=False)
        ].tolist()

        if duplicated:
            return AssertionResult(
                requirement=(
                    "X не должен содержать повторяющиеся "
                    "имена столбцов."
                ),
                requirement_source="входной контракт framework v1",
                expected="Повторяющиеся имена столбцов отсутствуют",
                actual=duplicated,
                status=CheckStatus.FAIL,
                evidence={
                    "повторяющиеся_столбцы": duplicated,
                },
                reason=(
                    "X содержит повторяющиеся имена столбцов."
                ),
            )

        return AssertionResult(
            requirement=(
                "X не должен содержать повторяющиеся "
                "имена столбцов."
            ),
            requirement_source="входной контракт framework v1",
            expected="Повторяющиеся имена столбцов отсутствуют",
            actual=[],
            status=CheckStatus.PASS,
            evidence={
                "повторяющиеся_столбцы": [],
            },
            reason=(
                "X не содержит повторяющихся имён столбцов."
            ),
        )

    @staticmethod
    def _check_metadata_vs_model_features(
        *,
        contract: ModelContract,
    ) -> AssertionResult:
        metadata_field = contract.required_features
        model_field = contract.model_input_features

        if metadata_field is None:
            return AssertionResult(
                requirement=(
                    "Контракт признаков из metadata должен "
                    "совпадать с model.feature_names_in_."
                ),
                requirement_source=(
                    "структурированная metadata + формальный API модели"
                ),
                expected=None,
                actual=None,
                status=CheckStatus.SKIP,
                reason=(
                    "Структурированная metadata не определяет "
                    "обязательные признаки."
                ),
            )

        if model_field is None:
            return AssertionResult(
                requirement=(
                    "Контракт признаков из metadata должен "
                    "совпадать с model.feature_names_in_."
                ),
                requirement_source=(
                    "структурированная metadata + формальный API модели"
                ),
                expected=list(metadata_field.value),
                actual=None,
                status=CheckStatus.SKIP,
                reason=(
                    "Модель не предоставляет формальную информацию "
                    "о входных признаках."
                ),
            )

        metadata_features = list(metadata_field.value)
        model_features = list(model_field.value)

        if metadata_features != model_features:
            missing_in_model = [
                feature
                for feature in metadata_features
                if feature not in model_features
            ]

            unexpected_in_model = [
                feature
                for feature in model_features
                if feature not in metadata_features
            ]

            return AssertionResult(
                requirement=(
                    "Контракт признаков из metadata должен "
                    "совпадать с model.feature_names_in_."
                ),
                requirement_source=(
                    f"{metadata_field.source} + "
                    f"{model_field.source}"
                ),
                expected=metadata_features,
                actual=model_features,
                status=CheckStatus.FAIL,
                evidence={
                    "отсутствуют_в_модели": missing_in_model,
                    "лишние_в_модели": unexpected_in_model,
                    "порядок_совпадает": (
                        metadata_features == model_features
                    ),
                },
                reason=(
                    "Структурированная metadata и формальная "
                    "информация о входных признаках модели "
                    "не совпадают."
                ),
            )

        return AssertionResult(
            requirement=(
                "Контракт признаков из metadata должен "
                "совпадать с model.feature_names_in_."
            ),
            requirement_source=(
                f"{metadata_field.source} + "
                f"{model_field.source}"
            ),
            expected=metadata_features,
            actual=model_features,
            status=CheckStatus.PASS,
            evidence={
                "отсутствуют_в_модели": [],
                "лишние_в_модели": [],
                "порядок_совпадает": True,
            },
            reason=(
                "Структурированная metadata и формальная "
                "информация о входных признаках модели совпадают."
            ),
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
                "X должен соответствовать формальному "
                "machine-readable входному контракту."
            ),
            requirement_source=(
                "структурированная metadata и формальный API модели"
            ),
            expected="Входной контракт соблюдён",
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
                "Все применимые проверки входного контракта пройдены."
            ),
            CheckStatus.WARN: (
                "Хотя бы одна применимая проверка входного "
                "контракта завершилась предупреждением."
            ),
            CheckStatus.FAIL: (
                "Хотя бы одна применимая проверка входного "
                "контракта не пройдена."
            ),
            CheckStatus.SKIP: (
                "Ни одна проверка входного контракта "
                "не могла быть выполнена."
            ),
        }

        return reasons[status]