from collections import defaultdict
from typing import Any, Literal

import networkx as nx
import numpy as np
import pyscipopt
from numpy.typing import NDArray


class BranchingTree:
    def __init__(self) -> None:
        self.tree = nx.DiGraph()

    def add_current_node_from_pyscipopt(
        self,
        model: pyscipopt.Model,
        step: int,
        action_set: NDArray[np.int64] | None,
        reward: float = 0.0,
        action: int = -1,
    ) -> None:
        node = model.getCurrentNode()
        if node is not None:
            node_id = node.getNumber()
            self.add_node(
                node_id,
                node,
                lower_bound=node.getLowerbound(),
                estimate=node.getEstimate(),
                step=step,
                action_set=action_set,
                visited=True,
            )
        open_leaves, open_children, open_siblings = model.getOpenNodes()
        for child in open_leaves:
            self.add_node(child.getNumber(), child, estimate=child.getEstimate())
        for child in open_children:
            self.add_node(child.getNumber(), child, estimate=child.getEstimate())
        for child in open_siblings:
            self.add_node(child.getNumber(), child, estimate=child.getEstimate())

    def add_node(
        self,
        node_id,
        node,
        lower_bound=None,
        estimate=None,
        step=None,
        action_set=None,
        visited=False,
        parent=None,
        parent_id=None,
    ) -> None:
        if node_id not in self.tree:
            self.tree.add_node(node_id)
        data = self.tree.nodes[node_id]
        if lower_bound is not None:
            data["lower_bound"] = lower_bound
        if estimate is not None:
            data["estimate"] = estimate
        if step is not None:
            data["step"] = step
        if action_set is not None:
            data["n_action_set"] = len(action_set)
        if visited is not None:
            data["visited"] = visited
        if parent is None:
            parent = node.getParent()
        if parent is not None:
            if parent_id is None:
                parent_id = parent.getNumber()
            if parent_id not in self.tree:
                self.add_node(parent_id, parent)
            self.tree.add_edge(parent_id, node_id)
        if self.tree.number_of_nodes() == 1:
            self.tree.graph["root"] = node_id

    def get_node_data(self, node_id) -> dict[str, Any] | None:
        """Returns the data of the node with the given ID."""
        if node_id in self.tree:
            return self.tree.nodes[node_id]
        return None

    def add_fathomed_nodes(self, nodes, status, visited) -> None:
        for node, node_id, parent, parent_id, lower_bound in nodes:
            if node_id not in self.tree:
                self.add_node(
                    node_id, node, visited=visited, parent=parent, parent_id=parent_id, lower_bound=lower_bound
                )
            self.tree.nodes[node_id]["status"] = status

    def add_infeasible_nodes(self, infeasible_nodes) -> None:
        self.add_fathomed_nodes(infeasible_nodes, "infeasible", visited=False)

    def add_feasible_nodes(self, feasible_nodes) -> None:
        self.add_fathomed_nodes(feasible_nodes, "feasible", visited=False)

    def add_solutions(self, solutions) -> None:
        for node_id, obj in solutions:
            if node_id == 0:
                continue
            if node_id not in self.tree:
                continue
            data = self.tree.nodes[node_id]
            data["obj"] = obj

    def _get_node_groups(self) -> dict:
        groups = defaultdict(lambda: [])
        for n in self.tree.nodes:
            node = self.tree.nodes[n]
            groups[node.get("status", "visited" if node.get("visited", False) else "unvisited")].append(n)
        return groups

    def _render_human(
        self,
        unvisited_node_colour: str = "#FFFFFF",
        visited_node_colour: str = "#A7C7E7",
        fathomed_node_colour: str = "#FF6961",
        incumbent_node_colour: str = "#C1E1C1",
        node_edge_colour: str = "#000000",
        use_latex_font: bool = False,
        font_scale: float = 0.75,
        context: Literal["paper", "notebook", "talk", "poster"] = "paper",
        style: Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] = "ticks",
        save_path: str | None = None,
        graphml_path: str | None = None,
    ) -> None:
        """Render the branch-and-bound search tree.

        Parameters
        ----------
        save_path : str or None
            If provided, save the figure to this path. Otherwise display interactively.
        graphml_path : str or None
            If provided, export the tree as GraphML to this path.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from networkx.drawing.nx_pydot import graphviz_layout
        except ImportError as e:
            raise ImportError(
                "Rendering requires optional dependencies: pip install gyozas[viz] or select ansi rendering"
            ) from e

        rc = {"text.usetex": True} if use_latex_font else {}
        font = "times" if use_latex_font else None
        sns.set_theme(font_scale=font_scale, context=context, style=style, rc=rc, font=font)  # pyright: ignore[reportArgumentType]

        group_to_colour = {
            "unvisited": unvisited_node_colour,
            "visited": visited_node_colour,
            "infeasible": fathomed_node_colour,
            "feasible": incumbent_node_colour,
        }

        _, _ = plt.subplots(figsize=(44, 10))

        if graphml_path is not None:
            nx.write_graphml(self.tree, graphml_path)

        pos = graphviz_layout(self.tree, prog="dot")

        node_groups = self._get_node_groups()
        for group_label, nodes in node_groups.items():
            nx.draw_networkx_nodes(
                self.tree,
                pos,
                nodelist=nodes,
                node_color=group_to_colour[group_label],
                edgecolors=node_edge_colour,
                label=group_label,
            )

        num_groups = len(node_groups)

        nx.draw_networkx_edges(self.tree, pos)

        nx.draw_networkx_labels(
            self.tree,
            pos,
            font_size=int(font_scale * 12),
            labels={node: f"{node}\n{self.tree.nodes[node].get('step', 'NA')}" for node in self.tree.nodes},
        )
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=num_groups)
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

    def _render_rich(self) -> None:
        """Render the tree using the ``rich`` library if available, else fall back to ANSI."""
        try:
            from rich import print as rprint
            from rich.tree import Tree as RichTree
        except ImportError:
            nx.write_network_text(self.tree)
            return

        _STATUS_STYLE = {
            "visited": "bold cyan",
            "unvisited": "dim white",
            "infeasible": "bold red",
            "feasible": "bold green",
        }

        def _label(node_id: int) -> str:
            data = self.tree.nodes[node_id]
            status = data.get(
                "status",
                "visited" if data.get("visited", False) else "unvisited",
            )
            style = _STATUS_STYLE.get(status, "white")
            parts = [f"[{style}]#{node_id}[/{style}]"]
            if "step" in data:
                parts.append(f"step={data['step']}")
            if "lower_bound" in data:
                parts.append(f"lb={data['lower_bound']:.3g}")
            if "estimate" in data:
                parts.append(f"est={data['estimate']:.3g}")
            if "obj" in data:
                parts.append(f"obj={data['obj']:.3g}")
            if "n_action_set" in data:
                parts.append(f"|A|={data['n_action_set']}")
            parts.append(f"[dim]({status})[/dim]")
            return "  ".join(parts)

        root_id = self.tree.graph.get("root")
        if root_id is None:
            if self.tree.number_of_nodes() == 0:
                rprint("[dim](empty tree)[/dim]")
                return
            # Fall back to any node with no incoming edges
            root_id = next(
                (n for n in self.tree.nodes if self.tree.in_degree(n) == 0),
                next(iter(self.tree.nodes)),
            )

        rich_tree = RichTree(_label(root_id))

        def _add_children(rich_node, node_id: int) -> None:
            for child_id in self.tree.successors(node_id):
                child_branch = rich_node.add(_label(child_id))
                _add_children(child_branch, child_id)

        _add_children(rich_tree, root_id)
        rprint(rich_tree)

    def render(self, render_mode) -> NDArray[np.uint8] | None:
        render_mode = render_mode.lower()
        if render_mode == "human":
            self._render_human()
        elif render_mode == "ansi":
            nx.write_network_text(self.tree)
        elif render_mode == "rich":
            self._render_rich()
        elif render_mode == "rgb_array":
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                from matplotlib.backends.backend_agg import FigureCanvasAgg
            except ImportError as e:
                raise ImportError(
                    "Rendering requires optional dependencies: pip install gyozas[viz] or select ansi rendering"
                ) from e

            nx.draw(self.tree)

            canvas = plt.get_current_fig_manager().canvas  # pyright: ignore[reportOptionalMemberAccess] # ty: ignore[unresolved-attribute]

            agg = canvas.switch_backends(FigureCanvasAgg)  # pyright: ignore[reportAttributeAccessIssue] # ty: ignore[unresolved-attribute]
            agg.draw()
            s, (width, height) = agg.print_to_buffer()

            X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
            return X[:, :, :3]
