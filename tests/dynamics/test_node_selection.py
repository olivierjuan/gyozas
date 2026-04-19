"""Unit tests for gyozas.dynamics.node_selection (construction only).

Note: Integration tests with full SCIP solve are omitted because
NodeSelectionDynamics can trigger segfaults with certain instances
in the pyscipopt C extension.
"""

from gyozas.dynamics import Dynamics
from gyozas.dynamics.node_selection import NodeSelectionDynamics


class TestConstruction:
    def test_is_dynamics_subclass(self):
        assert issubclass(NodeSelectionDynamics, Dynamics)

    def test_default_attributes(self):
        d = NodeSelectionDynamics()
        assert d.done is False
        assert d.nsteps == 0
        assert d.model is None

    def test_seed_does_not_raise(self):
        d = NodeSelectionDynamics()
        d.seed(123)
