from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from mlqa.context import ModelContract
from mlqa.contract import load_contract_from_metadata


@dataclass
class LoadedArtifacts:
    """
    Артефакты одного загруженного ML-пакета.

    Объект содержит модель, входные данные,
    необязательные эталонные значения и
    построенный machine-readable контракт.

    Новый вызов load_artifacts() создаёт новый объект,
    поэтому состояние предыдущей загрузки не переносится.
    """

    model: Any
    X: pd.DataFrame
    y: Any | None
    contract: ModelContract

    model_path: Path
    X_path: Path
    metadata_path: Path
    y_path: Path | None


def load_artifacts(
    *,
    model_path: str | Path,
    X_path: str | Path,
    metadata_path: str | Path,
    y_path: str | Path | None = None,
) -> LoadedArtifacts:
    """
    Загружает артефакты, необходимые framework v1.

    Обязательные файлы:
    - артефакт модели;
    - входные данные X;
    - структурированная metadata.

    y является необязательным.

    Функция не хранит глобальное состояние,
    не использует предыдущие результаты
    и не создаёт требования из X или y.
    """

    resolved_model_path = Path(model_path)
    resolved_X_path = Path(X_path)
    resolved_metadata_path = Path(metadata_path)

    resolved_y_path = (
        Path(y_path)
        if y_path is not None
        else None
    )

    _require_file(
        resolved_model_path,
        artifact_name="артефакт модели",
    )

    _require_file(
        resolved_X_path,
        artifact_name="входные данные X",
    )

    _require_file(
        resolved_metadata_path,
        artifact_name="структурированная metadata",
    )

    if resolved_y_path is not None:
        _require_file(
            resolved_y_path,
            artifact_name="эталонные значения y",
        )

    model = _load_model(resolved_model_path)

    X = _load_table(
        resolved_X_path,
        artifact_name="входные данные X",
    )

    contract = load_contract_from_metadata(
        metadata_path=resolved_metadata_path,
        model=model,
    )

    y = None

    if resolved_y_path is not None:
        y_table = _load_table(
            resolved_y_path,
            artifact_name="эталонные значения y",
        )

        y = _extract_target(
            y_table=y_table,
            contract=contract,
        )

    return LoadedArtifacts(
        model=model,
        X=X,
        y=y,
        contract=contract,
        model_path=resolved_model_path,
        X_path=resolved_X_path,
        metadata_path=resolved_metadata_path,
        y_path=resolved_y_path,
    )


def _require_file(
    path: Path,
    *,
    artifact_name: str,
) -> None:
    """
    Проверяет, что путь существует и указывает на файл.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл: {artifact_name}. "
            f"Путь: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Для артефакта «{artifact_name}» ожидался файл. "
            f"Путь: {path}"
        )


def _load_model(
    path: Path,
) -> Any:
    """
    Загружает сериализованный артефакт модели.

    Системные InconsistentVersionWarning от scikit-learn
    не выводятся пользователю. Совместимость версий
    проверяется отдельно в Check 5 и возвращается
    как структурированный ValidationResult.
    """

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                InconsistentVersionWarning,
            )

            return joblib.load(path)

    except Exception as exc:
        raise RuntimeError(
            "Не удалось загрузить артефакт модели. "
            f"Путь: {path}. "
            f"Ошибка: {type(exc).__name__}: {exc}"
        ) from exc


def _load_table(
    path: Path,
    *,
    artifact_name: str,
) -> pd.DataFrame:
    """
    Загружает поддерживаемый табличный файл.

    Framework v1 поддерживает:
    - .parquet
    - .csv
    """

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        reader = pd.read_parquet

    elif suffix == ".csv":
        reader = pd.read_csv

    else:
        raise ValueError(
            f"Неподдерживаемый формат файла для "
            f"артефакта «{artifact_name}»: "
            f"{suffix or '<без расширения>'}. "
            "Поддерживаются .parquet и .csv."
        )

    try:
        table = reader(path)

    except Exception as exc:
        raise RuntimeError(
            f"Не удалось загрузить {artifact_name}. "
            f"Путь: {path}. "
            f"Ошибка: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            f"Артефакт «{artifact_name}» после загрузки "
            "не является pandas.DataFrame."
        )

    return table


def _extract_target(
    *,
    y_table: pd.DataFrame,
    contract: ModelContract,
) -> Any | None:
    """
    Извлекает target из y только по формальному контракту.

    Имя target не определяется по содержимому y.
    Если target отсутствует в контракте или соответствующего
    столбца нет в y, ground truth считается недоступным.

    Зависимые проверки самостоятельно вернут SKIP.
    """

    if contract.target is None:
        return None

    target_name = contract.target.value

    if target_name not in y_table.columns:
        return None

    return y_table[target_name]