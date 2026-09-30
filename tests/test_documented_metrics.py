import numpy as np

from mlqa.checks.documented_metrics import DocumentedMetricsCheck
from mlqa.context import ContractValue, ModelContract
from mlqa.result import CheckStatus


class FixedProbabilityModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.7, 0.3],
                [0.1, 0.9],
            ]
        )


def make_contract(
    shipped_metrics=None,
):
    if shipped_metrics is None:
        shipped_metrics = {
            "ROC-AUC": 1.0,
            "PR-AUC": 1.0,
            "Brier": 0.0375,
        }

    return ModelContract(
        classes=ContractValue(
            value=[0, 1],
            source="model.classes_",
        ),
        shipped_metrics=ContractValue(
            value=shipped_metrics,
            source="model_metadata.json:test_metrics_shipped",
        ),
    )


def test_documented_metrics_pass_when_values_reproduce():
    X = [[10], [20], [30], [40]]
    y = np.array([0, 1, 0, 1])

    result = DocumentedMetricsCheck().run(
        contract=make_contract(),
        model=FixedProbabilityModel(),
        X=X,
        y=y,
    )

    assert result.status is CheckStatus.PASS
    assert len(result.assertions) == 3

    assert all(
        assertion.status is CheckStatus.PASS
        for assertion in result.assertions
    )


def test_documented_metrics_fail_when_value_does_not_reproduce():
    X = [[10], [20], [30], [40]]
    y = np.array([0, 1, 0, 1])

    contract = make_contract(
        shipped_metrics={
            "ROC-AUC": 0.5,
        }
    )

    result = DocumentedMetricsCheck().run(
        contract=contract,
        model=FixedProbabilityModel(),
        X=X,
        y=y,
    )

    assert result.status is CheckStatus.FAIL

    assertion = result.assertions[0]

    assert assertion.status is CheckStatus.FAIL
    assert assertion.expected == 0.5
    assert assertion.actual == 1.0
    assert assertion.evidence["метрика"] == "ROC-AUC"
    assert assertion.evidence["результат"] == "FAIL"


def test_documented_metrics_use_saved_decimal_precision():
    X = [[10], [20], [30], [40]]
    y = np.array([0, 1, 0, 1])

    contract = make_contract(
        shipped_metrics={
            "Brier": 0.0375,
        }
    )

    result = DocumentedMetricsCheck().run(
        contract=contract,
        model=FixedProbabilityModel(),
        X=X,
        y=y,
    )

    assert result.status is CheckStatus.PASS

    assertion = result.assertions[0]

    assert assertion.status is CheckStatus.PASS
    assert assertion.evidence["точность_сравнения"] == 4
    assert assertion.evidence[
        "получено_после_округления"
    ] == 0.0375


def test_documented_metrics_skip_metrics_without_v1_rule():
    X = [[10], [20], [30], [40]]
    y = np.array([0, 1, 0, 1])

    contract = make_contract(
        shipped_metrics={
            "ROC-AUC": 1.0,
            "Accuracy": 1.0,
            "Precision": 1.0,
            "Recall": 1.0,
            "F1": 1.0,
        }
    )

    result = DocumentedMetricsCheck().run(
        contract=contract,
        model=FixedProbabilityModel(),
        X=X,
        y=y,
    )

    assert result.status is CheckStatus.PASS

    statuses = {
        assertion.requirement: assertion.status
        for assertion in result.assertions
    }

    assert statuses[
        "Воспроизвести документированную метрику ROC-AUC."
    ] is CheckStatus.PASS

    assert statuses[
        "Воспроизвести документированную метрику Accuracy."
    ] is CheckStatus.SKIP

    assert statuses[
        "Воспроизвести документированную метрику Precision."
    ] is CheckStatus.SKIP

    assert statuses[
        "Воспроизвести документированную метрику Recall."
    ] is CheckStatus.SKIP

    assert statuses[
        "Воспроизвести документированную метрику F1."
    ] is CheckStatus.SKIP


def test_documented_metrics_skip_without_ground_truth():
    X = [[10], [20], [30], [40]]

    result = DocumentedMetricsCheck().run(
        contract=make_contract(),
        model=FixedProbabilityModel(),
        X=X,
        y=None,
    )

    assert result.status is CheckStatus.SKIP
    assert result.reason == (
        "Целевые значения y не предоставлены."
    )


def test_documented_metrics_skip_without_shipped_metrics():
    X = [[10], [20], [30], [40]]
    y = np.array([0, 1, 0, 1])

    contract = ModelContract(
        classes=ContractValue(
            value=[0, 1],
            source="model.classes_",
        )
    )

    result = DocumentedMetricsCheck().run(
        contract=contract,
        model=FixedProbabilityModel(),
        X=X,
        y=y,
    )

    assert result.status is CheckStatus.SKIP
    assert result.reason == (
        "В структурированных metadata "
        "не указаны документированные метрики."
    )


def test_documented_metrics_skip_without_probability_output():
    class ModelWithoutProbabilityOutput:
        pass

    X = [[10], [20], [30], [40]]
    y = np.array([0, 1, 0, 1])

    result = DocumentedMetricsCheck().run(
        contract=make_contract(),
        model=ModelWithoutProbabilityOutput(),
        X=X,
        y=y,
    )

    assert result.status is CheckStatus.SKIP

    applicable = [
        assertion
        for assertion in result.assertions
        if assertion.requirement.endswith(
            (
                "ROC-AUC.",
                "PR-AUC.",
                "Brier.",
            )
        )
    ]

    assert len(applicable) == 3

    assert all(
        assertion.status is CheckStatus.SKIP
        for assertion in applicable
    )


def test_documented_metrics_skip_without_formal_classes():
    X = [[10], [20], [30], [40]]
    y = np.array([0, 1, 0, 1])

    contract = ModelContract(
        shipped_metrics=ContractValue(
            value={
                "ROC-AUC": 1.0,
            },
            source="model_metadata.json:test_metrics_shipped",
        )
    )

    result = DocumentedMetricsCheck().run(
        contract=contract,
        model=FixedProbabilityModel(),
        X=X,
        y=y,
    )

    assert result.status is CheckStatus.SKIP

    assertion = result.assertions[0]

    assert assertion.status is CheckStatus.SKIP
    assert assertion.reason == (
        "Формальная информация о классах модели недоступна."
    )