from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ExperimentTracker(ABC):
    @abstractmethod
    def set_params(self, params: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def track(
        self,
        value: float,
        *,
        name: str,
        step: int,
        context: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_text(self, text: str, *, name: str, step: int = 0) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self, status: str = "completed") -> None:
        raise NotImplementedError
