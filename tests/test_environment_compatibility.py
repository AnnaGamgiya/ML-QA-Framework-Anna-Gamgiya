from unittest.mock import patch

from mlqa.checks.environment_compatibility import (
    EnvironmentCompatibilityCheck,
)
from mlqa.context import ContractValue, ModelContract
from mlqa.result import CheckStatus


def make_contract(
    library_versions=None,
):
    if library_versions is None:
        library_versions = {
            "sklearn": "1.2.3",
            "pandas": "2.0.0",
            "numpy": "1.26.0",
        }

    return ModelContract(
        library_versions=ContractValue(
            value=library_versions,
            source="model_metadata.json:library_versions",
        )
    )


def test_environment_pass_when_all_versions_match():
    installed_versions = {
        "scikit-learn": "1.2.3",
        "pandas": "2.0.0",
        "numpy": "1.26.0",
    }

    def fake_version(package_name):
        return installed_versions[package_name]

    with patch(
        "mlqa.checks.environment_compatibility.metadata.version",
        side_effect=fake_version,
    ):
        result = EnvironmentCompatibilityCheck().run(
            contract=make_contract(),
            model=None,
            X=None,
        )

    assert result.status is CheckStatus.PASS
    assert len(result.assertions) == 3

    assert all(
        assertion.status is CheckStatus.PASS
        for assertion in result.assertions
    )


def test_environment_warn_when_version_does_not_match():
    installed_versions = {
        "scikit-learn": "1.2.4",
        "pandas": "2.0.0",
        "numpy": "1.26.0",
    }

    def fake_version(package_name):
        return installed_versions[package_name]

    with patch(
        "mlqa.checks.environment_compatibility.metadata.version",
        side_effect=fake_version,
    ):
        result = EnvironmentCompatibilityCheck().run(
            contract=make_contract(),
            model=None,
            X=None,
        )

    assert result.status is CheckStatus.WARN

    sklearn_assertion = next(
        assertion
        for assertion in result.assertions
        if assertion.evidence["библиотека"] == "sklearn"
    )

    assert sklearn_assertion.status is CheckStatus.WARN
    assert sklearn_assertion.expected == "1.2.3"
    assert sklearn_assertion.actual == "1.2.4"
    assert sklearn_assertion.evidence["совпадение"] is False


def test_environment_mismatch_never_becomes_fail():
    contract = make_contract(
        library_versions={
            "numpy": "1.0.0",
        }
    )

    with patch(
        "mlqa.checks.environment_compatibility.metadata.version",
        return_value="9.9.9",
    ):
        result = EnvironmentCompatibilityCheck().run(
            contract=contract,
            model=None,
            X=None,
        )

    assert result.status is CheckStatus.WARN
    assert result.status is not CheckStatus.FAIL
    assert result.assertions[0].status is CheckStatus.WARN


def test_environment_skip_without_version_contract():
    result = EnvironmentCompatibilityCheck().run(
        contract=ModelContract(),
        model=None,
        X=None,
    )

    assert result.status is CheckStatus.SKIP
    assert result.reason == (
        "В структурированных metadata "
        "не указаны версии библиотек."
    )


def test_environment_skip_when_version_cannot_be_obtained():
    contract = make_contract(
        library_versions={
            "numpy": "1.26.0",
        }
    )

    with patch(
        "mlqa.checks.environment_compatibility.metadata.version",
        side_effect=(
            __import__("importlib").metadata.PackageNotFoundError
        ),
    ):
        result = EnvironmentCompatibilityCheck().run(
            contract=contract,
            model=None,
            X=None,
        )

    assert result.status is CheckStatus.SKIP

    assertion = result.assertions[0]

    assert assertion.status is CheckStatus.SKIP
    assert assertion.actual is None
    assert assertion.reason == (
        "Не удалось определить установленную "
        "версию пакета numpy."
    )


def test_environment_skip_unknown_library_mapping():
    contract = make_contract(
        library_versions={
            "example_library": "3.4.5",
        }
    )

    result = EnvironmentCompatibilityCheck().run(
        contract=contract,
        model=None,
        X=None,
    )

    assert result.status is CheckStatus.SKIP

    assertion = result.assertions[0]

    assert assertion.status is CheckStatus.SKIP
    assert assertion.evidence["библиотека"] == "example_library"
    assert assertion.expected == "3.4.5"


def test_environment_warn_has_priority_over_skipped_assertion():
    contract = make_contract(
        library_versions={
            "numpy": "1.26.0",
            "example_library": "3.4.5",
        }
    )

    with patch(
        "mlqa.checks.environment_compatibility.metadata.version",
        return_value="2.0.0",
    ):
        result = EnvironmentCompatibilityCheck().run(
            contract=contract,
            model=None,
            X=None,
        )

    assert result.status is CheckStatus.WARN

    statuses = {
        assertion.status
        for assertion in result.assertions
    }

    assert CheckStatus.WARN in statuses
    assert CheckStatus.SKIP in statuses