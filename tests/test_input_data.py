import pandas as pd

from mlqa.checks.input_data import InputDataCheck
from mlqa.context import ContractValue, ModelContract
from mlqa.result import CheckStatus


def make_contract() -> ModelContract:
    return ModelContract(
        required_features=ContractValue(
            value=["feature_a", "feature_b"],
            source="model_metadata.json",
        ),
        model_input_features=ContractValue(
            value=["feature_a", "feature_b"],
            source="model.feature_names_in_",
        ),
    )


def test_input_data_passes_when_contract_is_satisfied():
    X = pd.DataFrame(
        {
            "feature_a": [1, 2],
            "feature_b": [3, 4],
        }
    )

    result = InputDataCheck().run(
        contract=make_contract(),
        model=object(),
        X=X,
    )

    assert result.status is CheckStatus.PASS
    assert len(result.assertions) == 3
    assert all(
        assertion.status is CheckStatus.PASS
        for assertion in result.assertions
    )


def test_input_data_fails_when_required_feature_is_missing():
    X = pd.DataFrame(
        {
            "feature_a": [1, 2],
        }
    )

    result = InputDataCheck().run(
        contract=make_contract(),
        model=object(),
        X=X,
    )

    assert result.status is CheckStatus.FAIL

    required_features_assertion = result.assertions[0]

    assert required_features_assertion.status is CheckStatus.FAIL
    assert required_features_assertion.evidence == {
        "отсутствующие_признаки": ["feature_b"]
    }


def test_input_data_fails_on_duplicate_column_names():
    X = pd.DataFrame(
        [
            [1, 2],
            [3, 4],
        ],
        columns=["feature_a", "feature_a"],
    )

    contract = ModelContract(
        required_features=ContractValue(
            value=["feature_a"],
            source="model_metadata.json",
        )
    )

    result = InputDataCheck().run(
        contract=contract,
        model=object(),
        X=X,
    )

    assert result.status is CheckStatus.FAIL

    duplicate_assertion = result.assertions[1]

    assert duplicate_assertion.status is CheckStatus.FAIL
    assert duplicate_assertion.evidence == {
        "повторяющиеся_столбцы": [
            "feature_a",
            "feature_a",
        ]
    }


def test_input_data_fails_when_metadata_and_model_features_differ():
    X = pd.DataFrame(
        {
            "feature_a": [1, 2],
            "feature_b": [3, 4],
        }
    )

    contract = ModelContract(
        required_features=ContractValue(
            value=["feature_a", "feature_b"],
            source="model_metadata.json",
        ),
        model_input_features=ContractValue(
            value=["feature_a", "feature_c"],
            source="model.feature_names_in_",
        ),
    )

    result = InputDataCheck().run(
        contract=contract,
        model=object(),
        X=X,
    )

    assert result.status is CheckStatus.FAIL

    model_features_assertion = result.assertions[2]

    assert model_features_assertion.status is CheckStatus.FAIL
    assert model_features_assertion.evidence == {
        "отсутствуют_в_модели": ["feature_b"],
        "лишние_в_модели": ["feature_c"],
        "порядок_совпадает": False,
    }


def test_input_data_skips_assertions_without_feature_contract():
    X = pd.DataFrame(
        {
            "feature_a": [1, 2],
        }
    )

    result = InputDataCheck().run(
        contract=ModelContract(),
        model=object(),
        X=X,
    )

    assert result.status is CheckStatus.PASS

    assert result.assertions[0].status is CheckStatus.SKIP
    assert result.assertions[1].status is CheckStatus.PASS
    assert result.assertions[2].status is CheckStatus.SKIP


def test_input_data_does_not_fail_on_nan_without_explicit_requirement():
    X = pd.DataFrame(
        {
            "feature_a": [1.0, None],
            "feature_b": [None, 4.0],
        }
    )

    result = InputDataCheck().run(
        contract=make_contract(),
        model=object(),
        X=X,
    )

    assert result.status is CheckStatus.PASS


def test_input_data_does_not_treat_feature_role_as_dtype_contract():
    X = pd.DataFrame(
        {
            "feature_a": ["1", "2"],
            "feature_b": ["3", "4"],
        }
    )

    result = InputDataCheck().run(
        contract=make_contract(),
        model=object(),
        X=X,
    )

    assert result.status is CheckStatus.PASS