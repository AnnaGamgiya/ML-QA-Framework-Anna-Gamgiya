import json
from pathlib import Path
from typing import Any

from mlqa.context import ContractValue, ModelContract


def load_contract_from_metadata(
    metadata_path: str | Path,
    model: Any,
) -> ModelContract:
    """
    Создаёт ModelContract из структурированной metadata
    и формального API модели.

    X и y не используются как источники требований.
    """
    metadata_path = Path(metadata_path)

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    metadata_source = str(metadata_path)

    numeric_features = metadata.get(
        "numeric_features"
    )
    categorical_features = metadata.get(
        "categorical_features"
    )

    required_features = None

    if (
        numeric_features is not None
        and categorical_features is not None
    ):
        required_features = ContractValue(
            value=(
                list(numeric_features)
                + list(categorical_features)
            ),
            source=(
                f"{metadata_source}:"
                "numeric_features+categorical_features"
            ),
        )

    target = None

    if "target" in metadata:
        target = ContractValue(
            value=metadata["target"],
            source=f"{metadata_source}:target",
        )

    shipped_metrics = None

    if "test_metrics_shipped" in metadata:
        shipped_metrics = ContractValue(
            value=metadata[
                "test_metrics_shipped"
            ],
            source=(
                f"{metadata_source}:"
                "test_metrics_shipped"
            ),
        )

    library_versions = None

    if "library_versions" in metadata:
        library_versions = ContractValue(
            value=metadata[
                "library_versions"
            ],
            source=(
                f"{metadata_source}:"
                "library_versions"
            ),
        )

    model_input_features = None

    if hasattr(
        model,
        "feature_names_in_",
    ):
        model_input_features = ContractValue(
            value=list(
                model.feature_names_in_
            ),
            source="model.feature_names_in_",
        )

    classes = None

    if hasattr(
        model,
        "classes_",
    ):
        classes = ContractValue(
            value=list(
                model.classes_
            ),
            source="model.classes_",
        )

    prediction_capabilities = ContractValue(
        value={
            "predict": callable(
                getattr(
                    model,
                    "predict",
                    None,
                )
            ),
            "predict_proba": callable(
                getattr(
                    model,
                    "predict_proba",
                    None,
                )
            ),
        },
        source="формальный API модели",
    )

    return ModelContract(
        required_features=required_features,
        model_input_features=(
            model_input_features
        ),
        target=target,
        classes=classes,
        prediction_capabilities=(
            prediction_capabilities
        ),
        shipped_metrics=shipped_metrics,
        library_versions=library_versions,
    )