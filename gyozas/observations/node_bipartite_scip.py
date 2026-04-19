import numpy as np
from pyscipopt import Model

from gyozas.observations.structs import BipartiteGraph, EdgeFeatures


class NodeBipartiteSCIP:
    """Bipartite graph observation using PySCIPOpt's built-in C implementation.

    Returns the LP relaxation as a bipartite graph between constraint rows and
    variable columns, following Gasse et al. (NeurIPS 2019).
    """

    def __init__(self) -> None:
        pass

    def reset(self, model: Model) -> None:
        pass

    def extract(self, model: Model, done: bool) -> BipartiteGraph:
        obs = model.getBipartiteGraphRepresentation()
        variable_features = np.array(obs[0], dtype=np.float64)
        row_features = np.array(obs[2], dtype=np.float64)
        edge_indices = np.array([[x[1], x[0]] for x in obs[1]], dtype=np.int32).T
        edge_features = np.array([x[2] for x in obs[1]], dtype=np.float64)

        bg = BipartiteGraph(
            variable_features=variable_features,
            row_features=row_features,
            edge_features=EdgeFeatures(indices=edge_indices, values=edge_features),
        )
        return bg
