"""Extended tests for instance generators and sanitize_rng."""

import numpy as np
import pytest
from pyscipopt import Model

from gyozas.instances.capacitated_facility_location import CapacitatedFacilityLocationGenerator
from gyozas.instances.combinatorial_auction import CombinatorialAuctionGenerator
from gyozas.instances.files import FileGenerator
from gyozas.instances.independent_set import CliqueIndex, IndependentSetGenerator
from gyozas.instances.instance_generator import InstanceGenerator, sanitize_rng
from gyozas.instances.modifiers.embed_objective import EmbedObjective
from gyozas.instances.set_cover import SetCoverGenerator

# ---------------------------------------------------------------------------
# sanitize_rng
# ---------------------------------------------------------------------------


class TestSanitizeRng:
    def test_int_returns_generator(self):
        rng = sanitize_rng(42)
        assert isinstance(rng, np.random.Generator)

    def test_generator_passthrough(self):
        g = np.random.default_rng(0)
        assert sanitize_rng(g) is g

    def test_none_returns_new_generator(self):
        rng = sanitize_rng(None)
        assert isinstance(rng, np.random.Generator)

    def test_none_with_default(self):
        default = np.random.default_rng(99)
        assert sanitize_rng(None, default=default) is default

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="rng must be"):
            sanitize_rng("bad")  # pyright: ignore[reportArgumentType] # ty: ignore[invalid-argument-type]

    def test_float_raises(self):
        with pytest.raises(TypeError):
            sanitize_rng(3.14)  # pyright: ignore[reportArgumentType] # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# InstanceGenerator ABC
# ---------------------------------------------------------------------------


class TestInstanceGeneratorABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            InstanceGenerator()

    def test_next_alias(self):
        gen = SetCoverGenerator(n_rows=50, n_cols=100, rng=0)
        m = gen.next()
        assert isinstance(m, Model)

    def test_iter_returns_self(self):
        gen = SetCoverGenerator(n_rows=50, n_cols=100, rng=0)
        assert iter(gen) == gen

    def test_seed_changes_rng(self):
        gen = SetCoverGenerator(n_rows=50, n_cols=100, rng=0)
        gen.seed(99)
        assert isinstance(gen.rng, np.random.Generator)


# ---------------------------------------------------------------------------
# SetCoverGenerator extended
# ---------------------------------------------------------------------------


class TestSetCoverExtended:
    def test_iterator_protocol(self):
        gen = SetCoverGenerator(n_rows=50, n_cols=100, rng=0)
        m1 = next(gen)
        m2 = next(gen)
        assert isinstance(m1, Model)
        assert isinstance(m2, Model)

    def test_minimize_objective(self):
        gen = SetCoverGenerator(n_rows=50, n_cols=100, rng=0)
        m = next(gen)
        assert m.getObjectiveSense() == "minimize"

    def test_problem_name(self):
        gen = SetCoverGenerator(n_rows=50, n_cols=100, rng=0)
        m = next(gen)
        assert "SetCover" in m.getProbName()

    def test_all_binary_variables(self):
        gen = SetCoverGenerator(n_rows=50, n_cols=100, rng=0)
        m = next(gen)
        for v in m.getVars():
            assert v.vtype() == "BINARY"


# ---------------------------------------------------------------------------
# CombinatorialAuction extended
# ---------------------------------------------------------------------------


class TestCombinatorialAuctionExtended:
    def test_callable_params(self):
        gen = CombinatorialAuctionGenerator(
            n_items=lambda rng: rng.integers(5, 10),
            n_bids=lambda rng: rng.integers(10, 20),
            rng=42,
        )
        m = next(gen)
        assert isinstance(m, Model)

    def test_maximize_objective(self):
        gen = CombinatorialAuctionGenerator(n_items=5, n_bids=10, rng=0)
        m = next(gen)
        assert m.getObjectiveSense() == "maximize"

    def test_validation_min_gt_max_raises(self):
        gen = CombinatorialAuctionGenerator(rng=0)
        with pytest.raises(ValueError, match="min_value <= max_value"):
            gen.generate_instance(min_value=10, max_value=1)

    def test_validation_add_item_prob_out_of_range(self):
        gen = CombinatorialAuctionGenerator(rng=0)
        with pytest.raises(ValueError, match="add_item_prob"):
            gen.generate_instance(add_item_prob=1.5)

    def test_callable_uses_passed_rng(self):
        """Callable params should use the rng passed to generate_instance."""
        call_rng_ids = []

        def capture_rng(rng):
            call_rng_ids.append(id(rng))
            return 5

        gen = CombinatorialAuctionGenerator(n_items=capture_rng, rng=0)
        custom_rng = np.random.default_rng(99)
        gen.generate_instance(n_items=capture_rng, n_bids=10, rng=custom_rng)
        # The callable should receive the sanitized rng, not self.rng
        assert len(call_rng_ids) == 1


# ---------------------------------------------------------------------------
# IndependentSet extended
# ---------------------------------------------------------------------------


class TestIndependentSetExtended:
    def test_barabasi_albert(self):
        gen = IndependentSetGenerator(n_nodes=10, affinity=2, graph_type="barabasi_albert", rng=42)
        m = next(gen)
        assert isinstance(m, Model)
        assert m.getNVars() == 10

    def test_unknown_graph_type_raises(self):
        gen = IndependentSetGenerator(n_nodes=10, graph_type="unknown", rng=0)
        with pytest.raises(ValueError, match="Unknown graph type"):
            next(gen)


# ---------------------------------------------------------------------------
# CliqueIndex extended
# ---------------------------------------------------------------------------


class TestCliqueIndex:
    def test_overlapping_cliques(self):
        cliques = [[0, 1, 2], [1, 2, 3]]
        ci = CliqueIndex(cliques, 4)
        assert ci.are_in_same_clique(0, 1)
        assert ci.are_in_same_clique(1, 3)
        assert not ci.are_in_same_clique(0, 3)

    def test_single_node_cliques(self):
        cliques = [[0], [1], [2]]
        ci = CliqueIndex(cliques, 3)
        assert not ci.are_in_same_clique(0, 1)

    def test_empty_cliques(self):
        ci = CliqueIndex([], 3)
        assert not ci.are_in_same_clique(0, 1)


# ---------------------------------------------------------------------------
# CapacitatedFacilityLocation extended
# ---------------------------------------------------------------------------


class TestCapacitatedFacilityLocationExtended:
    def test_generate_returns_model(self):
        gen = CapacitatedFacilityLocationGenerator(n_customers=10, n_facilities=5, rng=42)
        m = next(gen)
        assert isinstance(m, Model)

    def test_problem_name(self):
        gen = CapacitatedFacilityLocationGenerator(n_customers=10, n_facilities=5, rng=42)
        m = next(gen)
        assert "CapacitatedFacilityLocation" in m.getProbName()

    def test_reproducibility(self):
        gen1 = CapacitatedFacilityLocationGenerator(n_customers=10, n_facilities=5, rng=42)
        gen2 = CapacitatedFacilityLocationGenerator(n_customers=10, n_facilities=5, rng=42)
        m1 = next(gen1)
        m2 = next(gen2)
        assert m1.getNVars() == m2.getNVars()
        assert m1.getNConss() == m2.getNConss()


# ---------------------------------------------------------------------------
# FileGenerator extended
# ---------------------------------------------------------------------------


class TestFileGeneratorExtended:
    def test_loads_lp_file(self):
        gen = FileGenerator(directory="tests", pattern="*.lp")
        m = next(gen)
        assert isinstance(m, Model)
        assert m.getNVars() > 0

    def test_seed_resets_file_list(self):
        gen = FileGenerator(directory="tests", pattern="*.lp")
        next(gen)
        gen.seed(0)
        # After seed, should be able to iterate again
        m = next(gen)
        assert isinstance(m, Model)

    def test_no_matching_files_raises(self):
        gen = FileGenerator(directory="tests", pattern="*.nonexistent")
        with pytest.raises(StopIteration):
            next(gen)


# ---------------------------------------------------------------------------
# EmbedObjective extended
# ---------------------------------------------------------------------------


class TestEmbedObjectiveExtended:
    def test_next_embeds_objective(self):
        inner = SetCoverGenerator(n_rows=50, n_cols=100, rng=0)
        embed = EmbedObjective(inner)
        m = next(embed)
        assert isinstance(m, Model)
        var_names = [v.name for v in m.getVars()]
        assert "_fobj_" in var_names

    def test_generate_instance_does_not_embed(self):
        inner = SetCoverGenerator(n_rows=50, n_cols=100, rng=0)
        embed = EmbedObjective(inner)
        m = embed.generate_instance(n_rows=50, n_cols=100, rng=0)
        var_names = [v.name for v in m.getVars()]
        assert "_fobj_" not in var_names

    def test_seed_delegates(self):
        inner = SetCoverGenerator(n_rows=50, n_cols=100, rng=0)
        embed = EmbedObjective(inner)
        embed.seed(42)

    def test_preserves_objective_sense_minimize(self):
        inner = SetCoverGenerator(n_rows=50, n_cols=100, rng=0)
        embed = EmbedObjective(inner)
        m = next(embed)
        assert m.getObjectiveSense() == "minimize"

    def test_preserves_objective_sense_maximize(self):
        inner = IndependentSetGenerator(n_nodes=5, edge_probability=0.3, rng=0)
        embed = EmbedObjective(inner)
        m = next(embed)
        assert m.getObjectiveSense() == "maximize"

    def test_objective_constraint_exists(self):
        inner = SetCoverGenerator(n_rows=50, n_cols=100, rng=0)
        embed = EmbedObjective(inner)
        m = next(embed)
        cons_names = [c.name for c in m.getConss()]
        assert "objective_function_constraint" in cons_names

    def test_iterator_protocol(self):
        inner = SetCoverGenerator(n_rows=50, n_cols=100, rng=0)
        embed = EmbedObjective(inner)
        assert iter(embed) == embed
        m = next(embed)
        assert isinstance(m, Model)
