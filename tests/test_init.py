"""Tests for gyozas.__init__ public API exports."""

import gyozas


class TestPublicAPI:
    def test_version_is_string(self):
        assert isinstance(gyozas.__version__, str)

    def test_all_exports_importable(self):
        for name in gyozas.__all__:
            assert hasattr(gyozas, name), f"{name} not in gyozas namespace"

    def test_key_classes_present(self):
        assert gyozas.Environment is not None
        assert gyozas.BranchingDynamics is not None
        assert gyozas.NodeSelectionDynamics is not None
        assert gyozas.GymnasiumWrapper is not None
        assert gyozas.SetCoverGenerator is not None
        assert gyozas.CombinatorialAuctionGenerator is not None
        assert gyozas.IndependentSetGenerator is not None
        assert gyozas.CapacitatedFacilityLocationGenerator is not None
        assert gyozas.FileGenerator is not None

    def test_observation_classes(self):
        assert gyozas.NodeBipartite is not None
        assert gyozas.NodeBipartiteSCIP is not None
        assert gyozas.NodeBipartiteEcole is not None
        assert gyozas.MetaObservation is not None

    def test_reward_classes(self):
        assert gyozas.NNodes is not None
        assert gyozas.Done is not None
        assert gyozas.SolvingTime is not None
        assert gyozas.LPIterations is not None
        assert gyozas.DualIntegral is not None
        assert gyozas.PrimalIntegral is not None
        assert gyozas.PrimalDualIntegral is not None

    def test_dynamics_classes(self):
        assert gyozas.Dynamics is not None
        assert gyozas.ExtraBranchingActions is not None

    def test_information_and_instance(self):
        assert gyozas.Empty is not None
        assert gyozas.InstanceGenerator is not None
