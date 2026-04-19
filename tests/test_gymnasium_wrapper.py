import pytest

from gyozas.gymnasium_wrapper import GymnasiumWrapper
from gyozas.instances.files import FileGenerator
from gyozas.rewards.nnodes import NNodes


@pytest.fixture
def wrapper():
    instances = FileGenerator(directory="tests", pattern="*.lp", rng=42)
    return GymnasiumWrapper(
        instance_generator=instances,
        reward_function=NNodes(),
        scip_params={"display/verblevel": 0, "limits/nodes": 50},
    )


class TestGymnasiumWrapper:
    def test_reset_returns_obs_and_info(self, wrapper):
        obs, info = wrapper.reset()
        assert "action_set" in info
        assert "reward_offset" in info
        wrapper.close()

    def test_full_episode(self, wrapper):
        obs, info = wrapper.reset()
        action_set = info["action_set"]
        steps = 0
        terminated = False
        while not terminated and action_set is not None:
            obs, reward, terminated, truncated, info = wrapper.step(0)
            action_set = info["action_set"]
            steps += 1
        assert steps > 0
        wrapper.close()

    def test_step_without_reset_raises(self, wrapper):
        with pytest.raises(RuntimeError):
            wrapper.step(0)

    def test_invalid_action_raises(self, wrapper):
        wrapper.reset()
        with pytest.raises(ValueError):
            wrapper.step(999999)
        wrapper.close()

    def test_max_steps_truncation(self):
        instances = FileGenerator(directory="tests", pattern="*.lp", rng=42)
        wrapper = GymnasiumWrapper(
            instance_generator=instances,
            reward_function=NNodes(),
            scip_params={"display/verblevel": 0, "limits/nodes": 500},
            max_steps=2,
        )
        obs, info = wrapper.reset()
        action_set = info["action_set"]
        if action_set is not None:
            obs, reward, terminated, truncated, info = wrapper.step(0)
            if not terminated and info["action_set"] is not None:
                obs, reward, terminated, truncated, info = wrapper.step(0)
                if not terminated:
                    assert truncated
        wrapper.close()

    def test_seed_via_reset(self, wrapper):
        obs1, info1 = wrapper.reset(seed=42)
        wrapper.close()
