"""RetroBinaryFathomed — RetroBranching's retro_binary_fathomed reward (R5.4)."""

import pytest

from gyozas.environment import Environment
from gyozas.instances import SetCoverGenerator
from gyozas.rewards import RetroBinaryFathomed, subtree_transitions


#: The official gasse_2019 SCIP parameters — crucially presolving/maxrestarts=0:
#: a SCIP restart renumbers the search tree, and the retro reward (like the
#: official implementation) would then orphan the experiences taken before it.
_GASSE_PARAMS = {
    "separating/maxrounds": 0,
    "presolving/maxrestarts": 0,
    "limits/time": 3600,
}


def _run_episode(node_limit=25, seed=7):
    """One mid-branching episode on the default set cover, node-limited."""
    model = SetCoverGenerator(rng=seed).generate_instance()
    model.hideOutput()
    reward_fn = RetroBinaryFathomed()
    env = Environment(
        iter([model]),
        reward_function=reward_fn,
        scip_params={"display/verblevel": 0, "limits/nodes": node_limit, **_GASSE_PARAMS},
    )
    obs, action_set, reward, done, info = env.reset()
    step_scores = []
    while not done:
        obs, action_set, reward, done, info = env.step(int(action_set[len(action_set) // 2]))
        step_scores.append(reward)
    return reward_fn, step_scores


class TestRetroBinaryFathomed:
    def test_per_step_scores_are_binary(self):
        """Each step scores the previous decision: -1 (open) or 0 (fathomed)."""
        _, step_scores = _run_episode()
        assert len(step_scores) >= 1
        assert set(step_scores) <= {0.0, -1.0}
        # The final decision is closed by construction (tree not updated at done).
        assert step_scores[-1] == 0.0

    def test_subtree_episodes_partition_the_decisions(self):
        """Every branching decision lands in exactly one sub-tree episode."""
        reward_fn, step_scores = _run_episode()
        episodes = reward_fn.subtree_episodes
        assert episodes is not None and len(episodes) >= 1
        all_indices = [idx for episode in episodes for idx in episode]
        # Disjoint and covering: a partition of the decisions.
        assert len(all_indices) == len(set(all_indices))
        assert set(all_indices) == set(range(len(step_scores)))
        for episode in episodes:
            values = list(episode.values())
            assert len(values) >= 1
            assert set(values) <= {0.0, -1.0}
            # Along a root-to-leaf path, nodes are visited in increasing order.
            assert list(episode.keys()) == sorted(episode.keys())

    def test_presolved_instance_gives_single_zero_episode(self):
        """A pre-solved instance never reaches a branching decision."""
        model = SetCoverGenerator(rng=7).generate_instance(n_rows=40, n_cols=60, density=0.05)
        model.hideOutput()
        reward_fn = RetroBinaryFathomed()
        env = Environment(iter([model]), reward_function=reward_fn, scip_params={"display/verblevel": 0})
        obs, action_set, reward, done, info = env.reset()
        assert done
        assert reward_fn.subtree_episodes == [{0: 0}]


class TestSubtreeTransitions:
    def test_reward_next_state_and_done_assembly(self):
        """Rewards map onto transitions; next state is the next path node's state."""
        experiences = [
            {"prev_state": f"s{i}", "state": f"n{i}", "action": i, "done": False}
            for i in range(5)
        ]
        episodes = [{0: -1.0, 2: -1.0, 4: 0.0}]
        transitions = subtree_transitions(experiences, episodes)
        assert len(transitions) == 1
        episode = transitions[0]
        assert [t["reward"] for t in episode] == [-1.0, -1.0, 0.0]
        assert [t["prev_state"] for t in episode] == ["s0", "s2", "s4"]
        # CHANGE: the next state of a transition is the following path node's state.
        assert [t["state"] for t in episode] == ["s2", "s4", "n4"]
        # CHANGE: the final sub-tree step is terminal, the others are not.
        assert [t["done"] for t in episode] == [False, False, True]

    def test_original_experiences_not_mutated(self):
        experiences = [
            {"prev_state": "s0", "state": "n0", "done": False},
            {"prev_state": "s1", "state": "n1", "done": False},
        ]
        subtree_transitions(experiences, [{0: -1.0, 1: 0.0}])
        assert experiences[0]["state"] == "n0"
        assert experiences[0]["done"] is False
        assert "reward" not in experiences[0]

    def test_change2_drops_final_transition_when_rewards_equal(self):
        """The final transition is dropped when its reward equals the per-step one."""
        experiences = [
            {"prev_state": f"s{i}", "state": f"n{i}", "done": False} for i in range(2)
        ]
        transitions = subtree_transitions(experiences, [{0: -1.0, 1: -1.0}])
        assert len(transitions) == 1
        assert len(transitions[0]) == 1
        assert transitions[0][0]["reward"] == -1.0
        assert transitions[0][0]["done"] is False

    def test_single_transition_episode_is_kept(self):
        """A one-step sub-tree episode is always kept (per-step reward is None)."""
        experiences = [{"prev_state": "s0", "state": "n0", "done": False}]
        transitions = subtree_transitions(experiences, [{0: -1.0}])
        assert len(transitions) == 1
        assert len(transitions[0]) == 1
        assert transitions[0][0]["done"] is True

    def test_real_episode_transitions(self):
        """End-to-end: episode experiences assembled from a real solve."""
        model = SetCoverGenerator(rng=7).generate_instance()
        model.hideOutput()
        reward_fn = RetroBinaryFathomed()
        env = Environment(
            iter([model]),
            reward_function=reward_fn,
            scip_params={"display/verblevel": 0, "limits/nodes": 25, **_GASSE_PARAMS},
        )
        obs, action_set, reward, done, info = env.reset()
        states = [id(obs)]  # placeholder states; only identity matters here
        while not done:
            obs, action_set, reward, done, info = env.step(int(action_set[len(action_set) // 2]))
            states.append(id(obs) if obs is not None else None)
        # One experience per decision: prev_state = state it acted on.
        n_decisions = len(states) - 1
        experiences = [
            {
                "prev_state": states[i],
                "state": states[i + 1],
                "action": i,
                "done": False,
            }
            for i in range(n_decisions)
        ]
        transitions = subtree_transitions(experiences, reward_fn.subtree_episodes)
        assert len(transitions) == len(reward_fn.subtree_episodes)
        for episode, episode_map in zip(transitions, reward_fn.subtree_episodes):
            assert len(episode) >= 1
            assert all(t["done"] is False for t in episode[:-1])
            if len(episode) == len(episode_map):
                # Not truncated by CHANGE #2: the final transition is terminal.
                assert episode[-1]["done"] is True
            else:
                # CHANGE #2 dropped the final transition (all rewards equal).
                assert len(episode) == len(episode_map) - 1
                assert episode[-1]["done"] is False
            for before, after in zip(episode, episode[1:]):
                assert before["state"] == after["prev_state"]
