from typing import Any

from mlqa.check import Check
from mlqa.context import ContractValue, ModelContract
from mlqa.result import (
    AssertionResult,
    CheckStatus,
    ValidationResult,
)
from mlqa.suite import QASuite


class DummyCheck(Check):
    @property
    def check_id(self) -> str:
        return "dummy"

    @property
    def check_name(self) -> str:
        return "Dummy check"

    def run(
        self,
        *,
        contract: ModelContract,
        model: Any,
        X: Any,
        y: Any | None = None,
    ) -> ValidationResult:
        assertion = AssertionResult(
            requirement="Dummy requirement",
            requirement_source="test",
            expected="expected",
            actual="expected",
            status=CheckStatus.PASS,
            evidence={"matched": True},
            reason="Requirement satisfied.",
        )

        return ValidationResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=CheckStatus.PASS,
            requirement=assertion.requirement,
            requirement_source=assertion.requirement_source,
            expected=assertion.expected,
            actual=assertion.actual,
            evidence=assertion.evidence,
            reason=assertion.reason,
            assertions=[assertion],
        )


def test_contract_preserves_value_and_source():
    contract = ModelContract(
        required_features=ContractValue(
            value=["feature_a", "feature_b"],
            source="model_metadata.json",
        )
    )

    field = contract.get("required_features")

    assert field is not None
    assert field.value == ["feature_a", "feature_b"]
    assert field.source == "model_metadata.json"
    assert contract.has("required_features") is True
    assert contract.has("target") is False


def test_contract_supports_explicit_extra_fields():
    contract = ModelContract(
        extra={
            "decision_threshold": ContractValue(
                value=0.155,
                source="model_metadata.json",
            )
        }
    )

    field = contract.get("decision_threshold")

    assert field is not None
    assert field.value == 0.155
    assert field.source == "model_metadata.json"


def test_validation_result_preserves_assertion_evidence():
    assertion = AssertionResult(
        requirement="Feature must exist",
        requirement_source="model_metadata.json",
        expected="distance_km present",
        actual="distance_km missing",
        status=CheckStatus.FAIL,
        evidence={"missing_features": ["distance_km"]},
        reason="Required feature is absent.",
    )

    result = ValidationResult(
        check_id="input_data",
        check_name="Input data",
        status=CheckStatus.FAIL,
        assertions=[assertion],
    )

    assert result.status is CheckStatus.FAIL
    assert len(result.assertions) == 1
    assert result.assertions[0].status is CheckStatus.FAIL
    assert result.assertions[0].evidence == {
        "missing_features": ["distance_km"]
    }


def test_suite_runs_checks_in_declared_order():
    first = DummyCheck()
    second = DummyCheck()

    suite = QASuite([first, second])

    results = suite.run(
        contract=ModelContract(),
        model=object(),
        X=object(),
    )

    assert len(results) == 2
    assert results[0].check_id == "dummy"
    assert results[1].check_id == "dummy"
    assert all(
        result.status is CheckStatus.PASS
        for result in results
    )