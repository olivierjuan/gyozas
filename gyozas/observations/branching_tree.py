import networkx as nx
import pyscipopt
from pyscipopt.scip import Model


class BranchingTreeObservation:
    def __init__(self) -> None:
        self.tree = nx.DiGraph()

    def reset(self, model: Model) -> None:
        self.tree = nx.DiGraph()

    def extract(self, model: Model, done: bool) -> nx.DiGraph:
        self._add_current_node_from_pyscipopt(model)
        return self.tree

    def _add_current_node_from_pyscipopt(self, model: pyscipopt.Model) -> None:
        node = model.getCurrentNode()
        if node is not None:
            node_id = node.getNumber()
            self._add_node(node_id, node, lower_bound=node.getLowerbound(), estimate=node.getEstimate(), visited=True)
        open_leaves, open_children, open_siblings = model.getOpenNodes()
        for child in open_leaves:
            self._add_node(child.getNumber(), child, estimate=child.getEstimate())
        for child in open_children:
            self._add_node(child.getNumber(), child, estimate=child.getEstimate())
        for child in open_siblings:
            self._add_node(child.getNumber(), child, estimate=child.getEstimate())

    def _add_node(
        self, node_id, node, lower_bound=None, estimate=None, visited=False, parent=None, parent_id=None
    ) -> None:
        if node_id not in self.tree:
            self.tree.add_node(node_id)
        data = self.tree.nodes[node_id]
        if lower_bound is not None:
            data["lower_bound"] = lower_bound
        if estimate is not None:
            data["estimate"] = estimate
        if visited is not None:
            data["visited"] = visited
        if parent is None:
            parent = node.getParent()
        if parent is not None:
            if parent_id is None:
                parent_id = parent.getNumber()
            if parent_id not in self.tree:
                self._add_node(parent_id, parent)
            self.tree.add_edge(parent_id, node_id)
        if self.tree.number_of_nodes() == 1:
            self.tree.graph["root"] = node_id
