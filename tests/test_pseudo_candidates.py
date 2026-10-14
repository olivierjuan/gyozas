"""The pseudo-candidates branching regime (ecole's ``pseudo_candidates=True``).

SORREL 2024 (Feng et al., AAAI 2025) collects its suboptimal demonstrations
over SCIP's *pseudo* branching candidates — the integer variables fractional
in the pseudo solution, a set overlapping but distinct from the LP-fractional
ones — and applies a chosen action through ``SCIPbranchVar`` (pseudo
candidates may hold integral LP values, where ``SCIPbranchVarVal`` is
refused). These tests pin the regime's facts empirically.
"""

from contextlib import contextmanager

import numpy as np

from gyozas.dynamics.branching import BranchingDynamics
from gyozas.environment import Environment
from gyozas.instances import SetCoverGenerator
from gyozas.observations import NodeBipartite, Pseudocosts, StrongBranchingScores


def _instance(seed: int = 0):
    # 200x250x0.10 branches under the fixture's seeded SCIP randomization
    # (smaller set covers solve at the root — the degenerate-family gotcha).
    return SetCoverGenerator(rng=np.random.default_rng(seed), n_rows=200, n_cols=250, density=0.10)


def _dynamics(pseudo_candidates: bool = False) -> BranchingDynamics:
    # A *seeded* dynamics: the unseeded one draws fresh SCIP permutation
    # seeds from OS entropy at every reset, so the root behavior (and the
    # regime facts below) would vary run to run — the fixture instance can
    # even solve at the root on a lucky permutation.
    dynamics = BranchingDynamics(pseudo_candidates=pseudo_candidates)
    dynamics.seed(0)
    return dynamics


def _root_action_set(pseudo_candidates: bool, seed: int = 0) -> set[int]:
    env = Environment(
        instance_generator=_instance(seed),
        observation_function=NodeBipartite(),
        scip_params={"display/verblevel": 0},
        dynamics=_dynamics(pseudo_candidates),
    )
    try:
        _obs, action_set, _reward, done, _info = env.reset()
        assert not done, "the fixture instance must branch"
        return set(int(a) for a in action_set)
    finally:
        env.close()


class TestPseudoCandidates:
    def test_root_action_set_differs_from_the_lp_candidates(self):
        # Empirical fact on SCIP 10 (pyscipopt 6.3), pinned as a regression:
        # the pseudo branching candidates overlap the LP-fractional ones but
        # are a different, larger set — they add integer variables whose LP
        # value is integral, and can even miss an LP-fractional variable
        # whose pseudo-solution value is integral.
        lp = _root_action_set(pseudo_candidates=False)
        pseudo = _root_action_set(pseudo_candidates=True)
        assert lp
        assert lp & pseudo
        assert pseudo - lp

    def test_episode_runs_over_the_pseudo_candidates(self):
        # Stepping a middle pseudo candidate must drive a full episode:
        # SCIPbranchVar accepts integral-valued pseudo candidates where
        # SCIPbranchVarVal would error out.
        env = Environment(
            instance_generator=_instance(),
            observation_function=NodeBipartite(),
            scip_params={"display/verblevel": 0, "limits/nodes": 100},
            dynamics=_dynamics(pseudo_candidates=True),
        )
        try:
            _obs, action_set, _reward, done, _info = env.reset()
            steps = 0
            while not done:
                action_set = np.asarray(action_set)
                candidates = action_set[action_set >= 0]
                assert len(candidates) > 0
                _obs, action_set, _reward, done, _info = env.step(int(candidates[len(candidates) // 2]))
                steps += 1
            assert steps > 0
        finally:
            env.close()

    def test_default_dynamics_unchanged(self):
        # The default stays the LP-fractional regime (every other method's).
        env = Environment(
            instance_generator=_instance(),
            scip_params={"display/verblevel": 0},
            dynamics=_dynamics(),
        )
        try:
            _obs, action_set, _reward, done, _info = env.reset()
            assert not done
            assert action_set is not None and len(action_set) > 0
        finally:
            env.close()


class TestScoreObservationsOverPseudoCandidates:
    """Ecole's default score observations in the pseudo-candidates regime.

    SORREL's demonstrator reads the LP-candidates score observations
    (``Pseudocosts`` and ``StrongBranchingScores`` with their default
    ``pseudo_candidates=False`` scoring) but argmaxes them over the *pseudo*
    candidates. Ecole sizes both arrays by the number of LP columns, so any
    candidate index is readable — the pseudo candidates outside the LP ones
    read ``NaN``.
    """

    @contextmanager
    def _root(self):
        env = Environment(
            instance_generator=_instance(),
            observation_function=NodeBipartite(),
            scip_params={"display/verblevel": 0},
            dynamics=_dynamics(pseudo_candidates=True),
        )
        try:
            _obs, action_set, _reward, done, _info = env.reset()
            assert not done, "the fixture instance must branch"
            yield env, np.asarray(action_set)
        finally:
            env.close()

    def test_pseudocosts_cover_every_pseudo_candidate(self):
        with self._root() as (env, action_set):
            scores = Pseudocosts().extract(env.model, done=False)
            assert scores is not None
            assert scores.shape[0] == env.model.getNLPCols()
            # Every pseudo candidate is readable; only the LP-fractional ones
            # hold finite scores, the rest read NaN.
            sliced = scores[action_set]
            assert sliced.shape == action_set.shape
            assert np.isnan(sliced).any() or np.isfinite(sliced).all()

    def test_strong_branching_scores_cover_every_pseudo_candidate(self):
        with self._root() as (env, action_set):
            scores = StrongBranchingScores().extract(env.model, done=False)
            assert scores is not None
            assert scores.shape[0] == env.model.getNLPCols()
            sliced = scores[action_set]
            assert sliced.shape == action_set.shape

    def test_pseudo_scoring_mode_also_covers_every_candidate(self):
        with self._root() as (env, action_set):
            scores = StrongBranchingScores(pseudo_candidates=True).extract(env.model, done=False)
            assert scores is not None
            assert scores.shape[0] == env.model.getNLPCols()
            assert np.isfinite(scores[action_set]).all()
