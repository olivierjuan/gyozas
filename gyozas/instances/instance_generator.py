from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator
from pyscipopt import Model


def sanitize_rng(rng: Generator | int | None, default: Generator | None = None) -> Generator:
    """Ensure the rng is a numpy random generator."""
    if isinstance(rng, int):
        return np.random.default_rng(rng)
    elif isinstance(rng, np.random.Generator):
        return rng
    elif rng is None:
        if default is not None:
            return default
        return np.random.default_rng()
    else:
        raise TypeError("rng must be an int, a numpy random Generator, or None.")


class InstanceGenerator(ABC):
    """
    Abstract base class for instance generators.
    Subclasses must implement the `generate_instance` method.
    """

    def __init__(self, rng: Generator | int | None = None) -> None:
        self.rng = sanitize_rng(rng)

    @abstractmethod
    def generate_instance(self, *args, **kwargs) -> Model:
        """
        Generate an instance with the given parameters.
        Must be implemented by subclasses.
        """
        pass

    def seed(self, seed) -> None:
        """Set the random seed for the generator."""
        self.rng = sanitize_rng(seed)

    def __iter__(self) -> "InstanceGenerator":
        """Return the iterator object itself."""
        return self

    @abstractmethod
    def __next__(self) -> Model:
        """Return the next instance."""
        pass

    def next(self) -> Model:
        """Alias for __next__."""
        return self.__next__()
