from abc import ABC, abstractmethod
from typing import Any

from mlqa.context import ModelContract
from mlqa.result import ValidationResult


class Check(ABC):
    """
    Базовый класс для всех пяти QA-check v1.

    Каждый check получает:
    - нормализованный ModelContract;
    - модель;
    - X;
    - опциональный y.

    Конкретный check самостоятельно определяет
    применимость своих assertions.
    """

    @property
    @abstractmethod
    def check_id(self) -> str:
        """Стабильный идентификатор проверки."""
        raise NotImplementedError

    @property
    @abstractmethod
    def check_name(self) -> str:
        """Человекочитаемое название проверки."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        *,
        contract: ModelContract,
        model: Any,
        X: Any,
        y: Any | None = None,
    ) -> ValidationResult:
        """
        Выполняет проверку и возвращает один
        стандартизированный ValidationResult.
        """
        raise NotImplementedError