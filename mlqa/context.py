from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ContractValue:
    """
    Одно значение контракта с обязательным указанием источника.

    Framework не должен терять provenance требования:
    значение всегда хранится вместе с тем, откуда оно получено.
    """

    value: Any
    source: str


@dataclass
class ModelContract:
    """
    Нормализованный machine-readable контракт ML-модели.

    Контракт содержит только объективные сведения,
    полученные из структурированной metadata или
    формального API/metadata модели.

    X и y не используются как источники требований
    о самих себе.
    """

    required_features: ContractValue | None = None
    model_input_features: ContractValue | None = None
    target: ContractValue | None = None
    classes: ContractValue | None = None

    prediction_capabilities: ContractValue | None = None

    shipped_metrics: ContractValue | None = None
    library_versions: ContractValue | None = None

    extra: dict[str, ContractValue] = field(
        default_factory=dict
    )

    def get(self, name: str) -> ContractValue | None:
        """
        Возвращает поле контракта по имени.

        Сначала проверяются стандартные поля v1,
        затем дополнительные явно сохранённые поля.
        """

        if name in self.__dataclass_fields__ and name != "extra":
            return getattr(self, name)

        return self.extra.get(name)

    def has(self, name: str) -> bool:
        """
        Проверяет наличие объективного значения контракта.
        """

        return self.get(name) is not None