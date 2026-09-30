from importlib import metadata
from typing import Any

from mlqa.check import Check
from mlqa.context import ModelContract
from mlqa.result import (
    AssertionResult,
    CheckStatus,
    ValidationResult,
)


PACKAGE_NAME_MAP = {
    "sklearn": "scikit-learn",
    "scikit-learn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "numpy",
}


class EnvironmentCompatibilityCheck(Check):
    """
    Check 5 — совместимость окружения.

    Сравнивает версии библиотек, записанные
    в machine-readable контракте, с версиями
    библиотек текущего runtime-окружения.

    Совпадение версии -> PASS.
    Несовпадение версии -> WARN.
    Версию невозможно определить -> SKIP.

    Само несовпадение версии не является FAIL:
    оно не доказывает, что модель технически
    неработоспособна.
    """

    @property
    def check_id(self) -> str:
        return "environment_compatibility"

    @property
    def check_name(self) -> str:
        return "Совместимость окружения"

    def run(
        self,
        *,
        contract: ModelContract,
        model: Any,
        X: Any,
        y: Any | None = None,
    ) -> ValidationResult:
        versions_field = contract.library_versions

        if versions_field is None:
            return self._skip_result(
                "В структурированных metadata "
                "не указаны версии библиотек."
            )

        expected_versions = dict(
            versions_field.value
        )

        if not expected_versions:
            return self._skip_result(
                "В структурированных metadata "
                "нет требований к версиям библиотек."
            )

        assertions: list[AssertionResult] = []

        for library_name, expected_version in expected_versions.items():
            package_name = PACKAGE_NAME_MAP.get(
                library_name
            )

            if package_name is None:
                assertions.append(
                    AssertionResult(
                        requirement=(
                            f"Проверить версию библиотеки "
                            f"{library_name}."
                        ),
                        requirement_source=versions_field.source,
                        expected=str(expected_version),
                        actual=None,
                        status=CheckStatus.SKIP,
                        evidence={
                            "библиотека": library_name,
                            "ожидаемая_версия": str(
                                expected_version
                            ),
                            "фактическая_версия": None,
                            "совпадение": None,
                        },
                        reason=(
                            f"Для библиотеки {library_name} "
                            "в Check 5 v1 не определено "
                            "однозначное имя установленного пакета."
                        ),
                    )
                )
                continue

            actual_version = self._get_installed_version(
                package_name
            )

            if actual_version is None:
                assertions.append(
                    AssertionResult(
                        requirement=(
                            f"Версия библиотеки {library_name} "
                            "должна соответствовать версии, "
                            "записанной для артефакта модели."
                        ),
                        requirement_source=versions_field.source,
                        expected=str(expected_version),
                        actual=None,
                        status=CheckStatus.SKIP,
                        evidence={
                            "библиотека": library_name,
                            "пакет": package_name,
                            "ожидаемая_версия": str(
                                expected_version
                            ),
                            "фактическая_версия": None,
                            "совпадение": None,
                        },
                        reason=(
                            f"Не удалось определить установленную "
                            f"версию пакета {package_name}."
                        ),
                    )
                )
                continue

            expected_version_text = str(
                expected_version
            )
            actual_version_text = str(
                actual_version
            )

            versions_match = (
                actual_version_text
                == expected_version_text
            )

            status = (
                CheckStatus.PASS
                if versions_match
                else CheckStatus.WARN
            )

            assertions.append(
                AssertionResult(
                    requirement=(
                        f"Версия библиотеки {library_name} "
                        "должна соответствовать версии, "
                        "записанной для артефакта модели."
                    ),
                    requirement_source=versions_field.source,
                    expected=expected_version_text,
                    actual=actual_version_text,
                    status=status,
                    evidence={
                        "библиотека": library_name,
                        "пакет": package_name,
                        "ожидаемая_версия": (
                            expected_version_text
                        ),
                        "фактическая_версия": (
                            actual_version_text
                        ),
                        "совпадение": versions_match,
                    },
                    reason=(
                        f"Версия библиотеки {library_name} "
                        "совпадает с версией, записанной "
                        "для артефакта модели."
                        if versions_match
                        else
                        f"Версия библиотеки {library_name} "
                        "не совпадает с версией, записанной "
                        "для артефакта модели. Само несовпадение "
                        "версии не доказывает неработоспособность "
                        "модели."
                    ),
                )
            )

        return self._build_result(
            assertions
        )

    @staticmethod
    def _get_installed_version(
        package_name: str,
    ) -> str | None:
        try:
            return metadata.version(
                package_name
            )
        except metadata.PackageNotFoundError:
            return None

    def _skip_result(
        self,
        reason: str,
    ) -> ValidationResult:
        assertion = AssertionResult(
            requirement=(
                "Версии библиотек текущего окружения "
                "должны быть сопоставлены с версиями, "
                "записанными для артефакта модели."
            ),
            requirement_source="структурированные metadata",
            expected=(
                "Версии библиотек доступны для сравнения"
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
                "Версии библиотек текущего окружения "
                "должны соответствовать требованиям "
                "machine-readable контракта."
            ),
            requirement_source=(
                "структурированные metadata"
            ),
            expected=(
                "Все доступные версии библиотек "
                "совпадают с записанными версиями"
            ),
            actual=status.value,
            evidence={
                "количество_проверок": len(
                    assertions
                ),
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
                "Все доступные для проверки версии "
                "библиотек совпадают."
            ),
            CheckStatus.WARN: (
                "Хотя бы одна версия библиотеки "
                "не совпадает с записанной версией."
            ),
            CheckStatus.FAIL: (
                "Проверка совместимости окружения "
                "завершилась с ошибкой."
            ),
            CheckStatus.SKIP: (
                "Ни одну версию библиотеки "
                "невозможно проверить."
            ),
        }

        return reasons[status]