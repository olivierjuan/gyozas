"""Tests for the documented fidelity profiles (R5.4, spec checklist §3)."""

from gyozas.experts import RelpscostExpert, StrongBranchingExpert
from gyozas.fidelity import available_fidelity_profiles, get_fidelity_profile
from gyozas.fidelity_profiles import register_documented_profiles
from gyozas.observations import NodeBipartite

DOCUMENTED = [
    "gasse",
    "lookback",
    "treegate",
    "tbrant",
    "mamba",
    "tgppo",
    "retrobranching",
    "scavuzzo",
]


class TestDocumentedProfiles:
    def test_all_eight_documented_profiles_are_registered(self):
        assert set(DOCUMENTED) <= set(available_fidelity_profiles())

    def test_bipartite_regime_features_and_expert(self):
        for name in ("gasse", "lookback"):
            profile = get_fidelity_profile(name)
            assert isinstance(profile.observation(), NodeBipartite)
            assert isinstance(profile.expert(), StrongBranchingExpert)
            assert profile.training["regime"] == "bipartite-LP"
            assert profile.training["native_strate"] == "ecole synthetics"

    def test_lookback_is_gasse_plus_its_loss(self):
        lookback = get_fidelity_profile("lookback")
        assert lookback.training["loss"] == "second-choice target smoothing + PAT regularizer"
        assert lookback.training["model_selection"]
        # Same regime otherwise.
        assert lookback.observation is not None and lookback.expert is not None

    def test_solver_stats_regime_expert_is_relpscost(self):
        for name in ("treegate", "tbrant", "mamba"):
            profile = get_fidelity_profile(name)
            assert isinstance(profile.expert(), RelpscostExpert)
            assert profile.training["augmentation_grid"] or name == "tbrant"

    def test_tbrant_declares_its_native_observation(self):
        # The runner builds the policy's environment from the profile's
        # observation; a profile without one hands the policy None features
        # (the regression that killed the first tbrant smoke run).
        from gyozas.observations import BranchingStateObservation

        assert isinstance(get_fidelity_profile("tbrant").observation(), BranchingStateObservation)

    def test_treegate_declares_its_native_observation(self):
        # TreeGate parameterized the 25+8+53 triple first (T-BranT and TGPPO
        # reused it); its profile recharges the same observation.
        from gyozas.observations import BranchingStateObservation

        assert isinstance(get_fidelity_profile("treegate").observation(), BranchingStateObservation)

    def test_treegate_augmentation_grid_matches_the_paper(self):
        grid = get_fidelity_profile("treegate").training["augmentation_grid"]
        assert grid == {"s": [0, 1, 2, 3, 4], "k": [0, 1, 5, 10, 15]}

    def test_rl_regimes_have_no_expert(self):
        for name in ("tgppo", "retrobranching", "scavuzzo"):
            profile = get_fidelity_profile(name)
            assert profile.expert is None
            assert profile.training["regime"].startswith("RL")

    def test_retrobranching_declares_its_native_observation(self):
        # The 43-variable-feature observation (19 bipartite + 24 retro). The
        # gasse_2019 parameters stay training-side (the runner would apply
        # profile scip_params at evaluation, breaking comparability — the
        # active solver profile stays a runner/cell setting, like every
        # other profile here).
        from gyozas.observations import NodeBipartiteRetro

        profile = get_fidelity_profile("retrobranching")
        assert isinstance(profile.observation(), NodeBipartiteRetro)
        assert profile.scip_params == {}
        assert profile.training["reward"].startswith("retro_binary_fathomed")
        assert "no restarts" in profile.training["scip_protocol"]

    def test_documented_scip_protocols_are_recorded_where_the_paper_has_one(self):
        # The RL papers (RetroBranching, Scavuzzo) leave the SCIP column open in the spec.
        for name in ("gasse", "lookback", "treegate", "tbrant", "mamba", "tgppo"):
            profile = get_fidelity_profile(name)
            assert "6.0.1" in profile.training["scip_protocol"] or "8.0.4" in profile.training["scip_protocol"]

    def test_documented_profiles_can_be_reregistered(self):
        assert register_documented_profiles() == DOCUMENTED
