"""Tests for RL trajectory recording (R4.4)."""

import random

import numpy as np
import pytest

from gyozas.environment import Environment
from gyozas.instances.files import FileGenerator
from gyozas.trajectories import (
    Trajectory,
    TrajectoryRecorder,
    load_trajectories,
    retrospective_trajectories,
    save_trajectories,
)


def _run_episode(env, recorder=None, seed=42):
    random.seed(seed)
    obs, action_set, reward, done, info = env.reset()
    if recorder is not None:
        recorder.record_step(obs, None, reward, done)
    steps = 0
    while not done:
        action = int(action_set[len(action_set) // 2])
        obs, action_set, reward, done, info = env.step(action)
        steps += 1
        if recorder is not None:
            recorder.record_step(obs, action, reward, done)
    return steps


class TestTrajectory:
    def test_len(self):
        trajectory = Trajectory(actions=[1, 2, 3], rewards=[0.0, 0.0, 1.0], dones=[False, False, True])
        assert len(trajectory) == 3

    def test_cumulative_returns_identity(self):
        trajectory = Trajectory(rewards=[1.0, 2.0, 3.0], dones=[False] * 3)
        assert trajectory.cumulative_returns(gamma=1.0) == [6.0, 5.0, 3.0]

    def test_cumulative_returns_discount(self):
        trajectory = Trajectory(rewards=[1.0, 2.0, 3.0], dones=[False] * 3)
        returns = trajectory.cumulative_returns(gamma=0.5)
        assert returns[0] == pytest.approx(1.0 + 0.5 * 2.0 + 0.25 * 3.0)
        assert returns[2] == pytest.approx(3.0)

    def test_dict_roundtrip(self):
        trajectory = Trajectory(observations=[np.zeros(3)], actions=[1], rewards=[0.5], dones=[True])
        assert Trajectory.from_dict(trajectory.to_dict()) == trajectory


class TestTrajectoryRecorder:
    def test_records_full_episode(self):
        env = Environment(
            instance_generator=FileGenerator(directory="tests", pattern="*.lp"),
            scip_params={"display/verblevel": 0, "limits/nodes": 50},
        )
        recorder = TrajectoryRecorder()
        try:
            steps = _run_episode(env, recorder)
            trajectory = recorder.end_episode()
            assert len(recorder) == 1
            assert len(trajectory) == steps + 1  # one record per step, plus the reset
            assert not any(trajectory.dones[:-1])
            assert trajectory.dones[-1] is True
            assert len(trajectory.rewards) == len(trajectory.actions) == len(trajectory.observations)
        finally:
            env.close()

    def test_end_episode_without_steps_raises(self):
        recorder = TrajectoryRecorder()
        with pytest.raises(RuntimeError):
            recorder.end_episode()

    def test_save_load_roundtrip(self, tmp_path):
        env = Environment(
            instance_generator=FileGenerator(directory="tests", pattern="*.lp"),
            scip_params={"display/verblevel": 0, "limits/nodes": 50},
        )
        recorder = TrajectoryRecorder()
        try:
            _run_episode(env, recorder)
            recorder.end_episode()
        finally:
            env.close()
        path = str(tmp_path / "traj.pkl")
        recorder.save(path)
        loaded = TrajectoryRecorder.load(path)
        assert len(loaded) == len(recorder)
        assert loaded.trajectories[0].rewards == recorder.trajectories[0].rewards
        assert loaded.trajectories[0].actions == recorder.trajectories[0].actions
        assert list(load_trajectories(path))[0].dones == recorder.trajectories[0].dones

    def test_save_trajectories_free_function(self, tmp_path):
        trajectory = Trajectory(actions=[1], rewards=[0.0], dones=[True])
        path = str(tmp_path / "t.pkl")
        save_trajectories([trajectory], path)
        (loaded,) = load_trajectories(path)
        assert loaded.actions == [1]


class TestRetrospectiveTrajectories:
    def test_one_trajectory_per_visited_node(self):
        env = Environment(
            instance_generator=FileGenerator(directory="tests", pattern="*.lp"),
            render_mode="human",
            scip_params={"display/verblevel": 0, "limits/nodes": 50},
        )
        try:
            _run_episode(env)
            trajectories = retrospective_trajectories(env.branching_tree)
            assert len(trajectories) > 0
            # The root trajectory is the shortest one.
            shortest = min(len(t) for t in trajectories)
            assert shortest == 1
            # Each trajectory ends with done=True and holds the root as first state.
            for trajectory in trajectories:
                assert trajectory.dones[-1] is True
                assert "action" in trajectory.observations[0] or len(trajectory.observations) >= 1
        finally:
            env.close()

    def test_paths_grow_with_depth(self):
        env = Environment(
            instance_generator=FileGenerator(directory="tests", pattern="*.lp"),
            render_mode="human",
            scip_params={"display/verblevel": 0, "limits/nodes": 50},
        )
        try:
            _run_episode(env, seed=7)
            trajectories = retrospective_trajectories(env.branching_tree)
            lengths = sorted(len(t) for t in trajectories)
            assert lengths[0] == 1
            assert lengths == sorted(lengths)
        finally:
            env.close()

    def test_empty_tree(self):
        from gyozas.branching_tree import BranchingTree

        assert retrospective_trajectories(BranchingTree()) == []
