from mlqa.checks.prediction_execution import PredictionExecutionCheck
from mlqa.context import ModelContract
from mlqa.result import CheckStatus


class WorkingModel:
    def predict(self, X):
        return [0] * len(X)


class FailingModel:
    def predict(self, X):
        raise ValueError("Prediction failed")


class ModelWithoutPredict:
    pass


def test_prediction_execution_passes_when_model_processes_input():
    model = WorkingModel()
    X = [1, 2, 3]

    result = PredictionExecutionCheck().run(
        contract=ModelContract(),
        model=model,
        X=X,
    )

    assert result.status is CheckStatus.PASS
    assert len(result.assertions) == 1
    assert result.assertions[0].status is CheckStatus.PASS
    assert result.assertions[0].evidence == {
        "этап": "predict",
        "исключение": None,
    }


def test_prediction_execution_fails_when_model_raises_exception():
    model = FailingModel()
    X = [1, 2, 3]

    result = PredictionExecutionCheck().run(
        contract=ModelContract(),
        model=model,
        X=X,
    )

    assert result.status is CheckStatus.FAIL
    assert len(result.assertions) == 1

    assertion = result.assertions[0]

    assert assertion.status is CheckStatus.FAIL
    assert assertion.evidence == {
        "этап": "predict",
        "тип_исключения": "ValueError",
        "сообщение_исключения": "Prediction failed",
    }


def test_prediction_execution_skips_when_model_is_missing():
    result = PredictionExecutionCheck().run(
        contract=ModelContract(),
        model=None,
        X=[1, 2, 3],
    )

    assert result.status is CheckStatus.SKIP
    assert result.assertions[0].status is CheckStatus.SKIP
    assert result.reason == "Артефакт модели не предоставлен."


def test_prediction_execution_skips_when_input_is_missing():
    result = PredictionExecutionCheck().run(
        contract=ModelContract(),
        model=WorkingModel(),
        X=None,
    )

    assert result.status is CheckStatus.SKIP
    assert result.assertions[0].status is CheckStatus.SKIP
    assert result.reason == "Входные данные X не предоставлены."


def test_prediction_execution_skips_without_prediction_interface():
    result = PredictionExecutionCheck().run(
        contract=ModelContract(),
        model=ModelWithoutPredict(),
        X=[1, 2, 3],
    )

    assert result.status is CheckStatus.SKIP
    assert result.assertions[0].status is CheckStatus.SKIP
    assert result.reason == (
        "Модель не предоставляет доступный интерфейс predict()."
    )