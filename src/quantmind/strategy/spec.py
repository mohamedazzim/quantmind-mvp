from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Mapping


class StrategySpecError(ValueError):
    """Raised when a declarative strategy specification is invalid."""


def _validate_json_value(value: Any, path: str = "value") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StrategySpecError(f"{path} contains a non-finite float")
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _validate_json_value(item, f"{path}[{i}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise StrategySpecError(f"{path} contains a non-string key")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise StrategySpecError(f"{path} contains unsupported type {type(value).__name__}")


@dataclass(frozen=True)
class StrategySpec:
    """Declarative strategy definition compiled only through a whitelist."""

    strategy_version: str
    feature_version: str
    signal_name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.strategy_version.strip():
            raise StrategySpecError("strategy_version must not be empty")
        if not self.feature_version.strip():
            raise StrategySpecError("feature_version must not be empty")
        if not self.signal_name.strip():
            raise StrategySpecError("signal_name must not be empty")
        _validate_json_value(dict(self.parameters), "parameters")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "strategy_version": self.strategy_version,
            "feature_version": self.feature_version,
            "signal_name": self.signal_name,
            "parameters": json.loads(
                json.dumps(dict(self.parameters), sort_keys=True, separators=(",", ":"), default=list)
            ),
        }

    def logic_identity_dict(self) -> dict[str, Any]:
        """Canonical logic identity; strategy_version is metadata, not an identity input."""
        return {
            "feature_version": self.feature_version,
            "signal_name": self.signal_name,
            "parameters": json.loads(
                json.dumps(dict(self.parameters), sort_keys=True, separators=(",", ":"), default=list)
            ),
        }

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    def logic_canonical_json(self) -> str:
        return json.dumps(self.logic_identity_dict(), sort_keys=True, separators=(",", ":"))
