import numpy as np
from pyscipopt.scip import Model

from gyozas.dynamics.probing import ProbeLedger
from gyozas.observations.node_bipartite_ecole import _CSTE, NodeBipartiteEcole, _obj_l2_norm
from gyozas.observations.structs import BipartiteGraph

# SCIP_BRANCHDIR enum values accepted by getVarPseudocost.
_BRANCHDIR_DOWN = 0
_BRANCHDIR_UP = 1

# Per-variable probing features appended to the standard NodeBipartite block.
PROBING_FEATURES = ("n_evaluations", "last_eval_age", "pseudocost_down", "pseudocost_up", "pseudocost_score")


class NodeBipartiteProbing(NodeBipartiteEcole):
    """NodeBipartite observation augmented with per-variable probing features.

    Extends :class:`NodeBipartiteEcole` by appending a block of features tailored to
    ``ProbingDynamics`` — most importantly the number of times each variable has already
    been strong-branched this episode (its *reliability*), read from a shared
    :class:`~gyozas.dynamics.probing.ProbeLedger`. Share the dynamics' ledger so the
    observation reflects the agent's own probing history::

        dyn = ProbingDynamics()
        obs = NodeBipartiteProbing(ledger=dyn.ledger)
        env = Environment(gen, observation_function=obs, dynamics=dyn)

    The appended variable features (in :data:`PROBING_FEATURES` order) are:

    - ``n_evaluations``: times the variable was strong-branched this episode (ledger).
    - ``last_eval_age``: nodes elapsed since it was last evaluated (0 if never).
    - ``pseudocost_down`` / ``pseudocost_up``: current per-direction pseudocost values.
    - ``pseudocost_score``: SCIP's combined pseudocost branching score — the estimate the
      dynamics would fall back on if the variable is *not* probed.

    Parameters
    ----------
    ledger
        The :class:`ProbeLedger` shared with the ``ProbingDynamics`` instance.
    normalize
        If ``True`` (default), scale the appended features to roughly ``[0, 1]`` /
        objective-normalised magnitudes (counts and ages use a saturating transform with
        constant ``_CSTE``; pseudocost values are divided by the objective L2 norm). If
        ``False``, the raw values are emitted unchanged.
    **kwargs
        Forwarded to :class:`NodeBipartiteEcole` (e.g. ``cache``, feature lists).
    """

    def __init__(self, ledger: ProbeLedger, normalize: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ledger = ledger
        self.normalize = normalize

    @property
    def probing_feature_names(self) -> tuple[str, ...]:
        return PROBING_FEATURES

    def extract(
        self, model: Model, done: bool, prev_var_features=None, prev_row_features=None, prev_edge_features=None
    ) -> BipartiteGraph | None:
        bg = super().extract(model, done, prev_var_features, prev_row_features, prev_edge_features)
        if bg is None:
            return None
        block = self._probing_block(model)
        bg.variable_features = np.hstack([bg.variable_features, block])
        return bg

    def _probing_block(self, model: Model) -> np.ndarray:
        # Same variable ordering as NodeBipartiteEcole: model.getVars(transformed=True).
        vars_ = model.getVars(transformed=True)
        block = np.zeros((len(vars_), len(PROBING_FEATURES)), dtype=np.float64)

        current = model.getCurrentNode()
        node_num = current.getNumber() if current is not None else 0
        obj_norm = _obj_l2_norm(vars_) if self.normalize else 1.0
        n_nodes = float(model.getNNodes())

        for i, var in enumerate(vars_):
            idx = var.getIndex()
            count = self.ledger.count(idx)
            last_node = self.ledger.last_node(idx)
            age = float(node_num - last_node) if last_node >= 0 else 0.0
            solval = var.getCol().getPrimsol()
            pc_down = model.getVarPseudocost(var, _BRANCHDIR_DOWN)
            pc_up = model.getVarPseudocost(var, _BRANCHDIR_UP)
            pc_score = model.getVarPseudocostScore(var, solval)

            if self.normalize:
                n_eval = count / (count + _CSTE)
                age = age / (n_nodes + _CSTE)
                pc_down = pc_down / obj_norm
                pc_up = pc_up / obj_norm
                pc_score = pc_score / (obj_norm * obj_norm)
            else:
                n_eval = float(count)

            block[i] = (n_eval, age, pc_down, pc_up, pc_score)
        return block
