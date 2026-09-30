from pathlib import Path
from typing import Any

from mlqa.checks.documented_metrics import DocumentedMetricsCheck
from mlqa.checks.environment_compatibility import (
    EnvironmentCompatibilityCheck,
)
from mlqa.checks.input_data import InputDataCheck
from mlqa.checks.prediction_execution import PredictionExecutionCheck
from mlqa.checks.prediction_output import PredictionOutputCheck
from mlqa.context import ModelContract
from mlqa.loader import load_artifacts
from mlqa.result import ValidationResult
from mlqa.suite import QASuite


def build_default_suite() -> QASuite:
    """
    Создаёт стандартный набор проверок framework v1.

    Порядок фиксирован:
    1. входные данные;
    2. выполнение предсказания;
    3. результат предсказания;
    4. воспроизводимость документированных метрик;
    5. совместимость окружения.
    """
    return QASuite(
        checks=[
            InputDataCheck(),
            PredictionExecutionCheck(),
            PredictionOutputCheck(),
            DocumentedMetricsCheck(),
            EnvironmentCompatibilityCheck(),
        ]
    )


def run_validation(
    *,
    contract: ModelContract,
    model: Any,
    X: Any,
    y: Any | None = None,
) -> list[ValidationResult]:
    """
    Запускает все пять проверок framework v1
    для уже загруженных артефактов.

    Функция не создаёт требования самостоятельно:
    требования уже должны находиться в ModelContract.
    """
    suite = build_default_suite()

    return suite.run(
        contract=contract,
        model=model,
        X=X,
        y=y,
    )


def run_validation_from_files(
    *,
    model_path: str | Path,
    X_path: str | Path,
    metadata_path: str | Path,
    y_path: str | Path | None = None,
) -> list[ValidationResult]:
    """
    Загружает артефакты ML-пакета и запускает
    стандартные пять проверок framework v1.

    Обязательные файлы:
    - артефакт модели;
    - входные данные X;
    - структурированная metadata.

    y является необязательным.

    Каждый вызов загружает новый набор артефактов
    и строит новый ModelContract. Состояние предыдущего
    запуска не используется.
    """
    artifacts = load_artifacts(
        model_path=model_path,
        X_path=X_path,
        metadata_path=metadata_path,
        y_path=y_path,
    )

    return run_validation(
        contract=artifacts.contract,
        model=artifacts.model,
        X=artifacts.X,
        y=artifacts.y,
    )