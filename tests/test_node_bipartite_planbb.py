"""NodeBipartitePlanBB — PlanB&B's 43-variable-feature observation (R5.4)."""

import numpy as np

from gyozas.environment import Environment
from gyozas.instances import SetCoverGenerator
from gyozas.observations import (
    N_PLANBB_FEATURES,
    PLANBB_FEATURE_NAMES,
    NodeBipartite,
    NodeBipartitePlanBB,
    NodeBipartiteRetro,
)
from gyozas.observations.meta_observation import MetaObservation


def _fresh_instance():
    """A fresh model (one Model cannot drive two environments).

    The default 500x1000 set cover: small ones get pre-solved at reset and
    never reach a branching decision.
    """
    gen = SetCoverGenerator(rng=7)
    model = gen.generate_instance()
    model.hideOutput()
    return model


def _paired_observation():
    """One environment observing with NodeBipartite, the retro and the PlanB&B
    observation.

    The dynamics draws a fresh variable/constraint permutation at every reset,
    so two environments on the same instance are *not* comparable — all three
    observations must come from the same model state.
    """
    meta = MetaObservation(
        {"base": NodeBipartite(), "retro": NodeBipartiteRetro(), "planbb": NodeBipartitePlanBB()}
    )
    env = Environment(iter([_fresh_instance()]), observation_function=meta, scip_params={"display/verblevel": 0})
    obs, action_set, reward, done, info = env.reset()
    return env, obs, action_set, done


class TestFeatureLayout:
    def test_names_match_official_order(self):
        assert N_PLANBB_FEATURES == 24
        assert len(PLANBB_FEATURE_NAMES) == 24
        assert PLANBB_FEATURE_NAMES[0] == "dual_bound_frac_change"
        assert PLANBB_FEATURE_NAMES[9] == "num_siblings_frac"
        assert PLANBB_FEATURE_NAMES[-1] == "branchability"

    def test_43_columns(self):
        env, obs, action_set, done = _paired_observation()
        assert not done
        planbb = obs["planbb"]
        n_vars = planbb.variable_features.shape[0]
        assert planbb.variable_features.shape == (n_vars, 43)

    def test_planbb_block_identical_across_variables(self):
        _, obs, _, _ = _paired_observation()
        planbb = obs["planbb"].variable_features[:, 19:]
        assert planbb.shape[1] == 24
        assert np.all(planbb == planbb[0])

    def test_base_columns_equal_node_bipartite(self):
        """Same model state: the first 19 columns are exactly NodeBipartite's."""
        _, obs, _, _ = _paired_observation()
        np.testing.assert_allclose(
            obs["planbb"].variable_features[:, :19], obs["base"].variable_features
        )
        np.testing.assert_allclose(obs["planbb"].row_features, obs["base"].row_features)
        np.testing.assert_allclose(
            obs["planbb"].edge_features.indices, obs["base"].edge_features.indices
        )
        np.testing.assert_allclose(
            obs["planbb"].edge_features.values, obs["base"].edge_features.values
        )

    def test_shared_slots_equal_retro(self):
        """The first 19 extra features are the retro ones and the three
        best-sibling-relative bounds match retro's; the two is_best_sibling
        flags and the constant branchability are the official deviations.
        Compared after a few branchings so the best-sibling slots are
        non-trivial."""
        env, obs, action_set, done = _paired_observation()
        steps = 0
        while not done and steps < 5:
            planbb = obs["planbb"].variable_features[0, 19:]
            retro = obs["retro"].variable_features[0, 19:]
            np.testing.assert_allclose(planbb[:19], retro[:19])
            np.testing.assert_allclose(planbb[19:22], retro[21:24])
            assert planbb[22] in (0.0, 1.0)  # is_best_sibling
            assert planbb[23] == 1.0  # branchability placeholder
            obs, action_set, reward, done, info = env.step(int(action_set[len(action_set) // 2]))
            steps += 1


class TestFeatureSemantics:
    def _root_values(self):
        _, obs, _, _ = _paired_observation()
        names = dict(enumerate(PLANBB_FEATURE_NAMES))
        return {names[i]: v for i, v in enumerate(obs["planbb"].variable_features[0, 19:])}

    def test_root_guards(self):
        """At the root the best node/sibling/parent may be None; the guards must hold."""
        values = self._root_values()
        assert values["curr_primal_dual_bound_gap_frac"] >= 0
        for key in ("is_curr_node_best", "is_curr_node_parent_best", "is_best_sibling"):
            assert values[key] in (0.0, 1.0)
        # Root: depth 0, no parent branchings, branchable.
        assert values["curr_node_depth"] == 0.0
        assert values["parent_branching_changes_frac"] == 0.0
        assert values["branchability"] == 1.0

    def test_root_is_best_sibling(self):
        """The root has no parent: is_best_sibling is 1 there whatever the
        sibling statistics say."""
        values = self._root_values()
        assert values["is_best_sibling"] == 1.0

    def test_no_frac_change_at_root(self):
        """The initial bounds are captured at the first extraction: zero drift there."""
        values = self._root_values()
        assert values["dual_bound_frac_change"] == 0.0
        assert values["primal_bound_frac_change"] == 0.0

    def test_reset_clears_captured_bounds(self):
        obs_fn = NodeBipartitePlanBB()
        obs_fn.reset(_fresh_instance())
        assert obs_fn._init_dual_bound is None
        assert obs_fn._init_primal_bound is None

    def test_features_across_steps(self):
        """A few steps in: features stay finite and the layout holds."""
        env, obs, action_set, done = _paired_observation()
        steps = 0
        while not done and steps < 5:
            obs, action_set, reward, done, info = env.step(int(action_set[len(action_set) // 2]))
            steps += 1
        assert steps >= 1
        if obs is not None:
            assert obs["planbb"].variable_features.shape[1] == 43
            assert np.isfinite(obs["planbb"].variable_features).all()
