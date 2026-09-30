import numpy as np

from mlqa.checks.prediction_output import (
    PROBABILITY_SUM_TOLERANCE,
    PredictionOutputCheck,
)
from mlqa.context import ContractValue, ModelContract
from mlqa.result import CheckStatus


class ValidClassifier:
    def predict(self, X):
        return np.array([0, 1, 1])

    def predict_proba(self, X):
        return np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [0.1, 0.9],
            ]
        )


def make_contract():
    return ModelContract(
        classes=ContractValue(
            value=[0, 1],
            source="model.classes_",
        )
    )


def test_prediction_output_passes_for_valid_classifier_output():
    X = [[1], [2], [3]]

    result = PredictionOutputCheck().run(
        contract=make_contract(),
        model=ValidClassifier(),
        X=X,
    )

    assert result.status is CheckStatus.PASS
    assert all(
        assertion.status is CheckStatus.PASS
        for assertion in result.assertions
    )


def test_prediction_output_fails_when_prediction_count_is_wrong():
    class WrongPredictionCountModel:
        def predict(self, X):
            return np.array([0, 1])

    X = [[1], [2], [3]]

    result = PredictionOutputCheck().run(
        contract=make_contract(),
        model=WrongPredictionCountModel(),
        X=X,
    )

    assert result.status is CheckStatus.FAIL

    assertion = result.assertions[0]

    assert assertion.status is CheckStatus.FAIL
    assert assertion.expected == 3
    assert assertion.actual == 2
    assert assertion.evidence == {
        "ожидаемое_количество_строк": 3,
        "количество_предсказаний": 2,
    }


def test_prediction_output_fails_for_unknown_predicted_class():
    class UnknownClassModel:
        def predict(self, X):
            return np.array([0, 2, 1])

    X = [[1], [2], [3]]

    result = PredictionOutputCheck().run(
        contract=make_contract(),
        model=UnknownClassModel(),
        X=X,
    )

    assert result.status is CheckStatus.FAIL

    assertion = result.assertions[1]

    assert assertion.status is CheckStatus.FAIL
    assert assertion.evidence == {
        "недопустимые_предсказания": [2]
    }


def test_prediction_output_fails_for_invalid_probability_shape():
    class WrongProbabilityShapeModel:
        def predict(self, X):
            return np.array([0, 1, 1])

        def predict_proba(self, X):
            return np.array(
                [
                    [1.0],
                    [1.0],
                    [1.0],
                ]
            )

    X = [[1], [2], [3]]

    result = PredictionOutputCheck().run(
        contract=make_contract(),
        model=WrongProbabilityShapeModel(),
        X=X,
    )

    assert result.status is CheckStatus.FAIL

    column_assertion = result.assertions[3]

    assert column_assertion.status is CheckStatus.FAIL
    assert column_assertion.expected == 2
    assert column_assertion.actual == 1


def test_prediction_output_fails_for_non_finite_probability():
    class NonFiniteProbabilityModel:
        def predict(self, X):
            return np.array([0, 1, 1])

        def predict_proba(self, X):
            return np.array(
                [
                    [0.8, 0.2],
                    [np.nan, 0.7],
                    [0.1, 0.9],
                ]
            )

    X = [[1], [2], [3]]

    result = PredictionOutputCheck().run(
        contract=make_contract(),
        model=NonFiniteProbabilityModel(),
        X=X,
    )

    assert result.status is CheckStatus.FAIL

    finite_assertion = result.assertions[4]

    assert finite_assertion.status is CheckStatus.FAIL
    assert finite_assertion.evidence == {
        "индексы_неконечных_значений": [[1, 0]]
    }


def test_prediction_output_fails_for_probability_outside_range():
    class OutOfRangeProbabilityModel:
        def predict(self, X):
            return np.array([0, 1, 1])

        def predict_proba(self, X):
            return np.array(
                [
                    [0.8, 0.2],
                    [-0.1, 1.1],
                    [0.1, 0.9],
                ]
            )

    X = [[1], [2], [3]]

    result = PredictionOutputCheck().run(
        contract=make_contract(),
        model=OutOfRangeProbabilityModel(),
        X=X,
    )

    assert result.status is CheckStatus.FAIL

    range_assertion = result.assertions[5]

    assert range_assertion.status is CheckStatus.FAIL
    assert range_assertion.evidence == {
        "индексы_значений_вне_диапазона": [
            [1, 0],
            [1, 1],
        ]
    }


def test_prediction_output_fails_when_probability_rows_do_not_sum_to_one():
    class InvalidProbabilitySumModel:
        def predict(self, X):
            return np.array([0, 1, 1])

        def predict_proba(self, X):
            return np.array(
                [
                    [0.8, 0.2],
                    [0.3, 0.6],
                    [0.1, 0.9],
                ]
            )

    X = [[1], [2], [3]]

    result = PredictionOutputCheck().run(
        contract=make_contract(),
        model=InvalidProbabilitySumModel(),
        X=X,
    )

    assert result.status is CheckStatus.FAIL

    sum_assertion = result.assertions[6]

    assert sum_assertion.status is CheckStatus.FAIL
    assert sum_assertion.evidence["строки_с_нарушением"] == [1]
    assert (
        sum_assertion.evidence["абсолютный_допуск"]
        == PROBABILITY_SUM_TOLERANCE
    )


def test_probability_sum_uses_fixed_tolerance():
    class ToleranceModel:
        def predict(self, X):
            return np.array([0])

        def predict_proba(self, X):
            return np.array(
                [
                    [
                        0.5,
                        0.5 + PROBABILITY_SUM_TOLERANCE / 2,
                    ]
                ]
            )

    X = [[1]]

    result = PredictionOutputCheck().run(
        contract=make_contract(),
        model=ToleranceModel(),
        X=X,
    )

    assert result.status is CheckStatus.PASS


def test_prediction_output_skips_class_assertion_without_class_contract():
    class ModelWithoutClassContract:
        def predict(self, X):
            return np.array([10, 20])

    X = [[1], [2]]

    result = PredictionOutputCheck().run(
        contract=ModelContract(),
        model=ModelWithoutClassContract(),
        X=X,
    )

    assert result.status is CheckStatus.PASS
    assert result.assertions[0].status is CheckStatus.PASS
    assert result.assertions[1].status is CheckStatus.SKIP


def test_prediction_output_skips_when_prediction_execution_fails():
    class BrokenModel:
        def predict(self, X):
            raise ValueError("Prediction failed")

    result = PredictionOutputCheck().run(
        contract=make_contract(),
        model=BrokenModel(),
        X=[[1], [2]],
    )

    assert result.status is CheckStatus.SKIP
    assert result.reason == (
        "Предсказания не были успешно получены; "
        "Check 3 зависит от успешного результата Check 2."
    )