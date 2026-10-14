"""DifficultyAdaptive — TGPPO 2025's difficulty-adaptive H3 reward."""

import math

from gyozas.environment import Environment
from gyozas.instances import SetCoverGenerator
from gyozas.rewards import DifficultyAdaptive


def _configured(baseline_nodes=1000.0, baseline_gap=0.1, baseline_pdi=10.0, solver_status="optimal", time_limit=60.0):
    """A reward configured with a mid-difficulty baseline."""
    reward = DifficultyAdaptive()
    reward.configure(baseline_nodes, baseline_gap, baseline_pdi, solver_status, time_limit)
    reward.reset(None)
    return reward


class _FakeModel:
    """The slice of the pyscipopt Model API the reward reads."""

    def __init__(self, nodes=10, solving_time=1.0, gap=0.05, pdi=5.0, pb=-10.0, db=-12.0, status="optimal"):
        self._nodes = nodes
        self._solving_time = solving_time
        self._gap = gap
        self._pdi = pdi
        self._pb = pb
        self._db = db
        self._status = status

    def getNNodes(self):
        return self._nodes

    def getSolvingTime(self):
        return self._solving_time

    def getGap(self):
        return self._gap

    def getPrimalDualIntegral(self):
        return self._pdi

    def getPrimalbound(self):
        return self._pb

    def getDualbound(self):
        return self._db

    def getStatus(self):
        return self._status


class TestDifficultySchedule:
    def test_weights_are_normalized_and_sum_to_one(self):
        for baseline_nodes in (1.0, 10.0, 1e3, 1e6, 1e9):
            reward = _configured(baseline_nodes=baseline_nodes)
            assert sum(reward.w.values()) == 1.0

    def test_easy_instances_weight_nodes_hard_instances_progress(self):
        easy = _configured(baseline_nodes=1.0)
        hard = _configured(baseline_nodes=1e6)
        # The schedule's endpoints (before normalization): 0.55 -> 0.25 nodes,
        # 0.10 -> 0.30 gap, 0.05 -> 0.20 pdi.
        assert easy.w["nodes"] > hard.w["nodes"]
        assert easy.w["gap"] < hard.w["gap"]
        assert easy.w["pdi"] < hard.w["pdi"]

    def test_unconfigured_defaults_to_a_one_node_timelimit_baseline(self):
        reward = DifficultyAdaptive()
        reward.reset(None)
        assert reward.B == 1.0
        assert reward.solver_status == "timelimit"
        assert reward.time_limit == 3600.0

    def test_pace_exponent_shrinks_with_difficulty(self):
        """The expected-curve exponent is 0.9 at the easiest end, 0.4 at the
        hardest: at half the time budget a trivial baseline is still behind
        its curve while a hard one is far ahead of it."""
        easy = _configured(baseline_nodes=1.0)
        hard = _configured(baseline_nodes=1e6)
        assert easy._pace(nodes=0, tfrac=0.5) < hard._pace(nodes=0, tfrac=0.5)
        # Endpoints of the schedule, checked through the curve itself. The
        # exponent is 0.9 - 0.5 * log1p(B)/log1p(1e6): 0.4 at B = 1e6, and
        # 0.9 - 0.5*log(2)/log(1e6+1) at the trivial B = 1 (the schedule's
        # logB never reaches 0). Through the reward's tanh scale of 2.0.
        lo_exp = 0.9 - 0.5 * math.log1p(1.0) / math.log1p(1e6)
        assert easy._pace(0, 0.5) == math.tanh(2.0 * (1.0 * 0.5**lo_exp) / (1.0 * 0.5**lo_exp + 1.0))
        assert hard._pace(0, 0.5) == math.tanh(2.0 * (1e6 * 0.5**0.4) / (1e6 * 0.5**0.4 + 1.0))


class TestExtraction:
    def test_first_extraction_is_a_no_op(self):
        """The official scores the first decision at the *second* branching
        call; the extraction before any decision returns 0 and leaves the
        progress bookkeeping untouched."""
        reward = _configured()
        assert reward.extract(_FakeModel(), done=False) == 0.0
        # prev_* stay at their initial values, so the first real step reward
        # sees no gap/pdi history (r_progress contribution only from bounds).
        assert reward.prev_gap == float("inf")
        assert reward.prev_pdi == 0.0
        assert reward.prev_pb is None

    def test_step_rewards_are_clipped_to_unit_box(self):
        reward = _configured()
        reward.extract(_FakeModel(), done=False)  # consume the no-op
        for gap in (0.0, 1.0, 1e6):
            r = reward.extract(_FakeModel(gap=gap), done=False)
            assert -1.0 <= r <= 1.0

    def test_terminal_bonus_optimal(self):
        """optimal: 1 + 3 * speedup with speedup capped at 5."""
        reward = _configured(baseline_nodes=100.0)
        reward.extract(_FakeModel(), done=False)
        # 10 nodes for a 100-node baseline -> speedup 10 capped at 5.
        assert reward.extract(_FakeModel(nodes=10), done=True) == 1.0 + 3.0 * 5.0
        reward = _configured(baseline_nodes=100.0)
        reward.extract(_FakeModel(), done=False)
        # 50 nodes -> speedup 2.
        assert reward.extract(_FakeModel(nodes=50), done=True) == 1.0 + 3.0 * 2.0

    def test_terminal_bonus_infeasible_and_timelimit(self):
        reward = _configured(baseline_nodes=100.0)
        reward.extract(_FakeModel(), done=False)
        assert reward.extract(_FakeModel(nodes=50, status="infeasible"), done=True) == 0.8 + 2.0 * 2.0
        reward = _configured(baseline_nodes=100.0, baseline_gap=0.5, baseline_pdi=10.0, solver_status="timelimit")
        reward.extract(_FakeModel(), done=False)
        # timelimit: 0.5*speedup + 0.3*tanh(baseline_gap - gap) +
        # 0.2*tanh((baseline_pdi - pdi)/baseline_pdi)
        expected = 0.5 * 2.0 + 0.3 * math.tanh(0.5 - 0.05) + 0.2 * math.tanh((10.0 - 5.0) / 10.0)
        assert (
            reward.extract(
                _FakeModel(nodes=50, gap=0.05, pdi=5.0, status="timelimit"),
                done=True,
            )
            == expected
        )

    def test_terminal_bonus_other_status(self):
        reward = _configured(baseline_nodes=100.0)
        reward.extract(_FakeModel(), done=False)
        assert reward.extract(_FakeModel(nodes=50, status="nodelimit"), done=True) == 0.3 * 2.0


class TestEnvironment:
    def test_full_episode_against_the_harness(self):
        """The reward drives a real episode end to end: a leading no-op, step
        rewards in [-1, 1], and a terminal reward matching the official
        formula for the final status (here the node limit's catch-all
        ``0.3 * speedup``)."""
        model = SetCoverGenerator(rng=7).generate_instance()
        model.hideOutput()
        reward_fn = DifficultyAdaptive()
        reward_fn.configure(
            baseline_nodes=100.0, baseline_gap=0.5, baseline_pdi=10.0, solver_status="optimal", time_limit=10.0
        )
        env = Environment(
            iter([model]),
            reward_function=reward_fn,
            scip_params={
                "display/verblevel": 0,
                "separating/maxrounds": 0,
                "presolving/maxrestarts": 0,
                "limits/time": 10,
                "limits/nodes": 30,
            },
        )
        _obs, action_set, first, _done, _info = env.reset()
        assert first == 0.0
        rewards = []
        done = False
        while not done:
            _obs, action_set, reward, done, _info = env.step(int(action_set[len(action_set) // 2]))
            rewards.append(reward)
        assert len(rewards) >= 1
        assert all(-1.0 <= r <= 1.0 for r in rewards[:-1])
        # The terminal reward is whatever the official formula prescribes for
        # the status the solver actually stopped on.
        status = model.getStatus()
        speedup = min(100.0 / max(model.getNNodes(), 1), 5.0)
        if status == "optimal":
            expected = 1.0 + 3.0 * speedup
        elif status in ("infeasible", "unbounded"):
            expected = 0.8 + 2.0 * speedup
        elif status == "timelimit":
            gap_gain = math.tanh(0.5 - model.getGap())
            pdi_gain = math.tanh((10.0 - model.getPrimalDualIntegral()) / 10.0)
            expected = 0.5 * speedup + 0.3 * gap_gain + 0.2 * pdi_gain
        else:  # nodelimit and any other stop
            expected = 0.3 * speedup
        assert rewards[-1] == expected

    def test_reset_clears_the_first_extraction_flag(self):
        reward = _configured()
        reward.extract(_FakeModel(), done=False)
        reward.reset(None)
        assert reward.extract(_FakeModel(), done=False) == 0.0
