"""The single policy interface (R1.1/R1.5).

The harness drives the solver, never the inverse: a policy is anything that,
given the current observation and the admissible action set, returns one
action. Architecturally it can be anything — GCNN, transformer, solver-stats
scalars, tree-level — the harness only sees the observation it declared and
the action it returns.
"""

from typing import Any, Protocol, runtime_checkable

from numpy.typing import NDArray


@runtime_checkable
class Policy(Protocol):
    """The one interface every branching policy exposes.

    ``observation`` is whatever the environment's observation function produced
    (a ``BipartiteGraph``, solver-stats scalars, an RL observation, a composite
    of several views) and ``action_set`` the admissible action indices at the
    current node. The returned action must be a member of ``action_set``.
    """

    def select_action(self, observation: Any, action_set: NDArray) -> int: ...


class CallablePolicy:
    """A policy built from a plain function — the one-line adoption point (R1.5).

    A new policy is one callable (or one :class:`Policy` class) plus the
    observation it declares; no harness change.

    Examples
    --------
    >>> import numpy as np
    >>> import gyozas
    >>> most_fractional = gyozas.CallablePolicy(
    ...     lambda obs, action_set: action_set[int(np.argmax(obs.variable_features[action_set, 4]))]
    ... )
    """

    def __init__(self, fn) -> None:
        self.fn = fn

    def select_action(self, observation: Any, action_set: NDArray) -> int:
        return int(self.fn(observation, action_set))

    def __repr__(self) -> str:
        return f"CallablePolicy({self.fn!r})"
