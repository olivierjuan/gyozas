import pickle
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass
class Trajectory:
    """One episode as a common offline format: states, actions, rewards, dones.

    Observations are stored as produced by the environment (arrays, dicts, ...);
    the format is pickle-serialized so any consumer can load it without gyozas
    bookkeeping.
    """

    observations: list[Any] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)

    def cumulative_returns(self, gamma: float = 1.0) -> list[float]:
        """Retrospective return at each step: the discounted sum of future rewards.

        The RetroBranching/Scavuzzo training signal: the return of step t is
        computed *after* the episode, from the rewards that followed it.
        """
        returns = [0.0] * len(self.rewards)
        running = 0.0
        for t in reversed(range(len(self.rewards))):
            running = self.rewards[t] + gamma * running
            returns[t] = running
        return returns

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trajectory":
        return cls(
            observations=list(data["observations"]),
            actions=list(data["actions"]),
            rewards=list(data["rewards"]),
            dones=list(data["dones"]),
        )


def save_trajectories(trajectories: list[Trajectory], path: str) -> None:
    """Serialize trajectories to a pickle file for offline consumption."""
    with open(path, "wb") as f:
        pickle.dump([t.to_dict() for t in trajectories], f)


def load_trajectories(path: str) -> list[Trajectory]:
    """Load trajectories from a pickle file produced by :func:`save_trajectories`."""
    with open(path, "rb") as f:
        data = pickle.load(f)  # noqa: S301
    return [Trajectory.from_dict(d) for d in data]


def retrospective_trajectories(branching_tree) -> list[Trajectory]:
    """Reconstruct one retrospective trajectory per visited node from a branching tree.

    For every node visited by the policy, the trajectory is the root-to-node
    path: the node records along the way (actions and rewards as recorded by the
    dynamics). This is the RetroBranching/Scavuzzo mode: trajectories are read
    back from the solved tree instead of being streamed step by step.
    """
    tree = branching_tree.tree
    root = tree.graph.get("root")
    if root is None:
        return []
    trajectories = []
    for visited_node in [n for n, data in tree.nodes(data=True) if data.get("visited")]:
        path = nx.shortest_path(tree, root, visited_node)
        trajectory = Trajectory()
        for node_id in path:
            data = tree.nodes[node_id]
            has_decision = "action" in data
            trajectory.observations.append(data)
            if has_decision:
                trajectory.actions.append(data["action"])
                trajectory.rewards.append(float(data["reward"]))
                trajectory.dones.append(False)
            elif node_id == visited_node:
                trajectory.dones.append(True)
            else:
                trajectory.actions.append(None)
                trajectory.rewards.append(0.0)
                trajectory.dones.append(False)
        trajectory.dones[-1] = True
        trajectories.append(trajectory)
    return trajectories


def nx_shortest_path(tree, source, target) -> list:
    import networkx as nx

    return nx.shortest_path(tree, source, target)


class TrajectoryRecorder:
    """Collects RL trajectories step by step in the common format.

    Feed every environment transition to :meth:`record_step` and call
    :meth:`end_episode` when the episode terminates. Trajectories are kept in
    memory and can be serialized offline with :func:`save_trajectories`.
    """

    def __init__(self) -> None:
        self.trajectories: list[Trajectory] = []
        self._current: Trajectory | None = None

    def record_step(self, observation: Any, action: Any, reward: float, done: bool) -> None:
        if self._current is None:
            self._current = Trajectory()
        self._current.observations.append(observation)
        self._current.actions.append(action)
        self._current.rewards.append(float(reward))
        self._current.dones.append(done)

    def end_episode(self) -> Trajectory:
        if self._current is None:
            raise RuntimeError("No episode in progress: call record_step first.")
        trajectory = self._current
        self.trajectories.append(trajectory)
        self._current = None
        return trajectory

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self.trajectories)

    def __len__(self) -> int:
        return len(self.trajectories)

    def save(self, path: str) -> None:
        save_trajectories(self.trajectories, path)

    @classmethod
    def load(cls, path: str) -> "TrajectoryRecorder":
        recorder = cls()
        recorder.trajectories = load_trajectories(path)
        return recorder
