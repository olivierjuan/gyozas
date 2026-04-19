import networkx as nx
from pyscipopt import Model, quicksum

from gyozas.instances.instance_generator import InstanceGenerator, sanitize_rng


class IndependentSetGenerator(InstanceGenerator):
    """Generator for random Maximum Independent Set problem instances."""

    def __init__(self, n_nodes=500, edge_probability=0.25, affinity=4, graph_type="barabasi_albert", rng=None) -> None:
        super().__init__(rng=rng)
        self.n_nodes = n_nodes
        self.edge_probability = edge_probability
        self.affinity = affinity
        self.graph_type = graph_type

    def __next__(self) -> Model:
        return self.generate_instance(
            n_nodes=self.n_nodes,
            edge_probability=self.edge_probability,
            affinity=self.affinity,
            graph_type=self.graph_type,
            rng=self.rng,
        )

    def generate_instance(
        self, n_nodes=500, edge_probability=0.25, affinity=4, graph_type="barabasi_albert", rng=None
    ) -> Model:
        rng = sanitize_rng(rng, default=self.rng)

        graph = self._make_graph(n_nodes, edge_probability, affinity, graph_type, rng)
        model = Model(problemName=f"IndependentSet-{n_nodes}")
        model.setMaximize()

        b_vars = [
            model.addVar(vtype="B", name=f"n_{i}", lb=0.0, ub=1.0, obj=1.0) for i in range(graph.number_of_nodes())
        ]
        clique_partition = list(nx.find_cliques_recursive(graph))

        # Clique constraints
        for clique in clique_partition:
            model.addCons(quicksum(b_vars[n] for n in clique) <= 1.0, name=f"clique_{clique}")

        # Edge constraints for edges not covered by cliques
        clique_index = CliqueIndex(clique_partition, graph.number_of_nodes())
        for n1, n2 in graph.edges():
            if not clique_index.are_in_same_clique(n1, n2):
                model.addCons(b_vars[n1] + b_vars[n2] <= 1.0, name=f"edge_{n1}_{n2}")

        return model

    def _make_graph(self, n_nodes=50, edge_probability=0.1, affinity=2, graph_type="erdos_renyi", rng=None) -> nx.Graph:
        if rng is None:
            rng = self.rng
        if graph_type.lower() == "erdos_renyi":
            return nx.erdos_renyi_graph(n_nodes, edge_probability, seed=rng)
        elif graph_type.lower() == "barabasi_albert":
            m = min(affinity, n_nodes - 1)
            return nx.barabasi_albert_graph(n_nodes, m, seed=rng)
        else:
            raise ValueError("Unknown graph type")


class CliqueIndex:
    def __init__(self, cliques: list[list[int]], n_nodes: int) -> None:
        self.node_cliques: list[set[int]] = [set() for _ in range(n_nodes)]
        for clique_id, clique in enumerate(cliques):
            for node in clique:
                self.node_cliques[node].add(clique_id)

    def are_in_same_clique(self, n1, n2) -> bool:
        return len(self.node_cliques[n1] & self.node_cliques[n2]) > 0
