"""SubtreeSizes — Scavuzzo et al. 2022's tree-MDP credit assignment (rl2branch)."""

from gyozas.environment import Environment
from gyozas.instances import SetCoverGenerator
from gyozas.rewards import SubtreeSizes

#: The official gasse_2019 SCIP parameters — crucially presolving/maxrestarts=0:
#: a SCIP restart renumbers the search tree and orphans the recording.
_GASSE_PARAMS = {
    "separating/maxrounds": 0,
    "presolving/maxrestarts": 0,
    "limits/time": 3600,
}


def _run_episode(node_limit=25, seed=7):
    """One mid-branching episode on the default set cover, node-limited."""
    model = SetCoverGenerator(rng=seed).generate_instance()
    model.hideOutput()
    reward_fn = SubtreeSizes()
    env = Environment(
        iter([model]),
        reward_function=reward_fn,
        scip_params={"display/verblevel": 0, "limits/nodes": node_limit, **_GASSE_PARAMS},
    )
    obs, action_set, reward, done, info = env.reset()
    step_rewards = []
    while not done:
        obs, action_set, reward, done, info = env.step(int(action_set[len(action_set) // 2]))
        step_rewards.append(reward)
    return reward_fn, step_rewards


class _FakeNode:
    """The slice of the pyscipopt Node API the recorder reads."""

    def __init__(self, number, parent, depth, lowerbound=0.0):
        self._number = number
        self._parent = parent
        self._depth = depth
        self._lowerbound = lowerbound

    def getNumber(self):
        return self._number

    def getParent(self):
        return self._parent

    def getDepth(self):
        return self._depth

    def getLowerbound(self):
        return self._lowerbound


class _FakeModel:
    """Yields one focus node per extract call, as the branch-and-bound loop does."""

    def __init__(self, nodes):
        self._nodes = iter(nodes)

    def getCurrentNode(self):
        return next(self._nodes, None)


class TestRecording:
    def test_stepwise_reward_is_always_zero(self):
        """The reward is retrospective: per-step scores stay at 0 (the tree-MDP
        return is read off ``subtree_sizes`` by the training loop)."""
        _, step_rewards = _run_episode()
        assert len(step_rewards) >= 1
        assert set(step_rewards) == {0.0}

    def test_one_node_id_per_decision(self):
        """Every branching decision records exactly one focus node, in order."""
        reward_fn, step_rewards = _run_episode()
        assert len(reward_fn.node_ids) == len(step_rewards)
        assert len(set(reward_fn.node_ids)) == len(reward_fn.node_ids)

    def test_tree_shape(self):
        """Each decision has a recorded parent (except the first — the SCIP
        root), and its sub-tree size counts its children and all descendants."""
        reward_fn, _ = _run_episode()
        tree = reward_fn._tree
        assert len(tree) == len(reward_fn.node_ids)
        roots = [n for n, rec in tree.items() if rec["parent"] is None]
        assert len(roots) == 1
        assert roots[0] == reward_fn.node_ids[0]
        # Every other decision's parent is itself a recorded decision, and a
        # decision's sub-tree holds at least its two children.
        for node_id, record in tree.items():
            if record["parent"] is not None:
                assert record["parent"] in tree
                assert reward_fn.subtree_sizes[node_id] >= 2

    def test_root_subtree_covers_the_whole_tree(self):
        """The first decision's sub-tree holds every node below the SCIP root:
        2 children per decision."""
        reward_fn, _ = _run_episode()
        root = reward_fn.node_ids[0]
        assert reward_fn.subtree_sizes[root] == 2 * len(reward_fn.node_ids)

    def test_presolved_instance_records_nothing(self):
        """A pre-solved instance never reaches a branching decision."""
        model = SetCoverGenerator(rng=7).generate_instance(n_rows=40, n_cols=60, density=0.05)
        model.hideOutput()
        reward_fn = SubtreeSizes()
        env = Environment(iter([model]), reward_function=reward_fn, scip_params={"display/verblevel": 0})
        obs, action_set, reward, done, info = env.reset()
        assert done
        assert reward_fn.node_ids == []
        assert reward_fn.subtree_sizes == {}


class TestSubtreeSizes:
    def test_hand_built_tree(self):
        """A root with two internal children and one leaf child:
        sizes accumulate bottom-up over depth groups."""
        reward_fn = SubtreeSizes()
        # depth 0: node 1; depth 1: nodes 2, 3; depth 2: nodes 4, 5.
        nodes = [
            _FakeNode(1, None, 0),
            _FakeNode(2, _FakeNode(1, None, 0), 1),
            _FakeNode(3, _FakeNode(1, None, 0), 1),
            _FakeNode(4, _FakeNode(2, _FakeNode(1, None, 0), 1), 2),
            _FakeNode(5, _FakeNode(2, _FakeNode(1, None, 0), 1), 2),
        ]
        model = _FakeModel(nodes)
        for _ in range(5):
            assert reward_fn.extract(model, done=False) == 0.0
        reward_fn.extract(model, done=True)
        assert reward_fn.node_ids == [1, 2, 3, 4, 5]
        # Leaves 4, 5: two children each; node 3: two children; node 2: its two
        # children plus their four; node 1: everyone below it.
        assert reward_fn.subtree_sizes == {1: 10, 2: 6, 3: 2, 4: 2, 5: 2}

    def test_returns_are_minus_size_minus_one(self):
        """The tree-MDP return of a decision is -(subtree_size) - 1: closing a
        sub-tree immediately scores -3, every extra node costs one more."""
        reward_fn = SubtreeSizes()
        nodes = [_FakeNode(1, None, 0), _FakeNode(2, _FakeNode(1, None, 0), 1), _FakeNode(3, _FakeNode(1, None, 0), 1)]
        model = _FakeModel(nodes)
        for _ in range(3):
            reward_fn.extract(model, done=False)
        reward_fn.extract(model, done=True)
        returns = {n: -reward_fn.subtree_sizes[n] - 1 for n in reward_fn.node_ids}
        assert returns == {1: -7, 2: -3, 3: -3}

    def test_children_per_branch(self):
        """The recorder assumes binary branching (lp_cand=True in the official
        code) — parameterised here for the synthetic tree."""
        reward_fn = SubtreeSizes(children_per_branch=3)
        nodes = [_FakeNode(1, None, 0), _FakeNode(2, _FakeNode(1, None, 0), 1)]
        model = _FakeModel(nodes)
        for _ in range(2):
            reward_fn.extract(model, done=False)
        reward_fn.extract(model, done=True)
        assert reward_fn.subtree_sizes == {1: 6, 2: 3}

    def test_reset_clears_previous_episode(self):
        """A second episode starts from an empty tree."""
        reward_fn = SubtreeSizes()
        model = _FakeModel([_FakeNode(1, None, 0), _FakeNode(2, _FakeNode(1, None, 0), 1)])
        reward_fn.extract(model, done=False)
        reward_fn.extract(model, done=False)
        reward_fn.extract(model, done=True)
        assert reward_fn.subtree_sizes == {1: 4, 2: 2}
        reward_fn.reset(None)
        model = _FakeModel([_FakeNode(1, None, 0)])
        reward_fn.extract(model, done=False)
        reward_fn.extract(model, done=True)
        assert reward_fn.node_ids == [1]
        assert reward_fn.subtree_sizes == {1: 2}
