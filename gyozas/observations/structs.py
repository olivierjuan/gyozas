from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pyscipopt import Model


@runtime_checkable
class ObservationFunction(Protocol):
    def reset(self, model: Model) -> None: ...
    def extract(self, model: Model, done: bool) -> Any: ...


@dataclass
class EdgeFeatures:
    indices: np.ndarray
    values: np.ndarray


@dataclass
class BipartiteGraph:
    variable_features: np.ndarray
    row_features: np.ndarray
    edge_features: EdgeFeatures
