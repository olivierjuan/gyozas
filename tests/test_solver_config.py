"""Tests for declarative solver configuration (R2.2)."""

import pytest
from pyscipopt import Model

from gyozas.solver_config import SolverConfig, available_profiles, get_profile, register_profile


class TestSolverConfig:
    def test_apply_sets_params(self):
        config = SolverConfig(name="test", params={"display/verblevel": 0, "limits/nodes": 42})
        model = Model()
        config.apply(model)
        assert model.getParam("limits/nodes") == 42
        assert model.getParam("display/verblevel") == 0

    def test_apply_does_not_mutate_config(self):
        config = SolverConfig(name="test", params={"limits/nodes": 42})
        model = Model()
        config.apply(model)
        assert config.params == {"limits/nodes": 42}

    def test_to_dict_roundtrip(self):
        config = SolverConfig(name="proto", params={"limits/time": 3600.0, "display/verblevel": 0})
        data = config.to_dict()
        assert data == {"name": "proto", "params": {"limits/time": 3600.0, "display/verblevel": 0}}
        restored = SolverConfig.from_dict(data)
        assert restored == config

    def test_equality_by_value(self):
        assert SolverConfig(name="a", params={"x": 1}) == SolverConfig(name="a", params={"x": 1})


class TestProfiles:
    def test_default_profile_registered_and_empty(self):
        profile = get_profile("default")
        assert profile.params == {}
        assert "default" in available_profiles()

    def test_register_and_get_profile(self):
        profile = SolverConfig(name="lin2022-test", params={"limits/time": 3600.0})
        register_profile(profile)
        assert get_profile("lin2022-test") is profile

    def test_get_unknown_profile_raises(self):
        with pytest.raises(KeyError, match="no-such-profile"):
            get_profile("no-such-profile")

    def test_lin2022_profile_applies(self):
        profile = get_profile("lin2022")
        model = Model()
        profile.apply(model)
        assert model.getParam("limits/time") == 3600.0
        assert model.getParam("limits/nodes") == -1
        assert model.getParam("conflict/usesb") is False
        assert model.getParam("heuristics/feaspump/freq") == -1
        assert model.getParam("heuristics/locks/freq") == -1

    def test_lin2022_profile_disables_all_heuristics(self):
        model = Model()
        profile = get_profile("lin2022")
        profile.apply(model)
        for name in profile.params:
            if name.startswith("heuristics/"):
                assert model.getParam(name) == -1, name

    def test_apply_with_cutoff_sets_objlimit(self):
        profile = SolverConfig(name="cutoff-test", params={"display/verblevel": 0})
        model = Model()
        profile.apply(model, cutoff=123.0)
        assert model.getObjlimit() == 123.0
