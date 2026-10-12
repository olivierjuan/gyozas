"""Tests for the SolveStatus information function (R2.5)."""

from pyscipopt import Model

from gyozas.environment import Environment
from gyozas.informations import SolveStatus
from gyozas.instances.files import FileGenerator


def _tiny_model() -> Model:
    model = Model()
    model.setParam("display/verblevel", 0)
    x = model.addVar(vtype="BINARY", name="x")
    y = model.addVar(vtype="BINARY", name="y")
    model.setObjective(x + y, "maximize")
    return model


class TestSolveStatusUnit:
    def test_optimal_is_solved_not_truncated(self):
        model = _tiny_model()
        model.optimize()
        info = SolveStatus().extract(model, done=True)
        assert info["status"] == "optimal"
        assert info["solved"] is True
        assert info["truncated"] is False
        assert info["limit"] is None

    def test_time_limit_is_truncated_at_time(self):
        model = _tiny_model()
        model.setParam("limits/time", 0.0)
        model.optimize()
        info = SolveStatus().extract(model, done=True)
        assert info["status"] == "timelimit"
        assert info["solved"] is False
        assert info["truncated"] is True
        assert info["limit"] == "time"

    def test_node_limit_is_truncated_at_nodes(self):
        model = Model()
        model.setParam("display/verblevel", 0)
        model.setParam("limits/nodes", 1)
        model.readProblem("tests/instance.lp")
        model.optimize()
        info = SolveStatus().extract(model, done=True)
        assert info["status"] == "nodelimit"
        assert info["truncated"] is True
        assert info["limit"] == "nodes"

    def test_infeasible_counts_as_solved(self):
        model = _tiny_model()
        z = model.addVar(vtype="BINARY", name="z")
        model.addCons(z >= 2)
        model.optimize()
        info = SolveStatus().extract(model, done=True)
        assert info["status"] == "infeasible"
        assert info["solved"] is True
        assert info["truncated"] is False


class TestSolveStatusInEnvironment:
    def test_end_of_episode_always_reports(self):
        env = Environment(
            instance_generator=FileGenerator(directory="tests", pattern="*.lp"),
            information_function=SolveStatus(),
            scip_params={"display/verblevel": 0, "limits/nodes": 50},
        )
        try:
            obs, action_set, reward, done, info = env.reset()
            while not done:
                obs, action_set, reward, done, info = env.step(action_set[len(action_set) // 2])
            assert set(info) == {"status", "solved", "truncated", "limit"}
            assert isinstance(info["status"], str)
            assert isinstance(info["solved"], bool)
            assert isinstance(info["truncated"], bool)
            assert (info["limit"] is None) == (not info["truncated"])
        finally:
            env.close()
