import warnings

import numpy as np
from numpy.typing import NDArray
from pyscipopt.scip import PY_SCIP_STAGE, Column, Model, Variable

from gyozas.observations.structs import BipartiteGraph, EdgeFeatures

_CSTE = 5.0  # age scaling constant (matches Ecole)

# ---------------------------------------------------------------------------
# Default feature lists (NodeBipartitePy mode)
# ---------------------------------------------------------------------------
_DEFAULT_STATIC_COL = ("obj_coef", "binary", "integer", "implicit_integer", "continuous")
_DEFAULT_DYNAMIC_COL = (
    "has_lb",
    "has_ub",
    "red_cost",
    "sol_val",
    "sol_frac",
    "sol_at_lb",
    "sol_at_ub",
    "age",
    "best_incumbent_val",
    "avg_incumbent_val",
    "basis_lower",
    "basis_basic",
    "basis_upper",
    "basis_zero",
)
_DEFAULT_STATIC_ROW = ("bias", "obj_cosine")
_DEFAULT_DYNAMIC_ROW = ("is_tight", "dual_sol", "age")

# ---------------------------------------------------------------------------
# _put — feature assignment helper
# ---------------------------------------------------------------------------


def _put(row, fm: dict, name: str, value) -> None:
    """Assign value to row[fm[name]] if name is a requested feature."""
    if name in fm:
        row[fm[name]] = value


# ---------------------------------------------------------------------------
# Behavioural helpers
# ---------------------------------------------------------------------------


def _obj_l2_norm(vars_) -> float:
    norm = float(np.linalg.norm([v.getObj() for v in vars_]))
    return norm if norm > 0.0 else 1.0


def _get_vars_and_cols(model: Model) -> list[tuple[Variable, Column]]:
    """Return list of (var, col) pairs to iterate over."""
    return [(v, v.getCol()) for v in model.getVars(transformed=True)]


def _edge_col_index(col: Column) -> int:
    """Column index used in the edge sparse matrix."""
    return col.getLPPos()


def _sol_frac(model: Model, col: Column) -> float:
    """Feasibility fraction"""
    return model.feasFrac(col.getPrimsol())


def _incumbent_val(model: Model, var: Variable) -> float:
    sol = model.getBestSol()
    if sol is None:
        return 0.0
    return model.getSolVal(sol, var)


def _avg_incumbent_val(model: Model, var: Variable) -> float:
    if model.getBestSol() is None:
        return 0.0
    return var.getAvgSol()


def _obj_cosine(model: Model, row, obj_norm: float) -> float:
    norm_prod = row.getNorm() * obj_norm
    if model.isGT(norm_prod, 0.0):
        return model.getRowObjParallelism(row)
    return 0.0


# ---------------------------------------------------------------------------
# Per-variable feature filling
# ---------------------------------------------------------------------------


def _fill_static_var(row, var, obj_norm: float, fm: dict) -> None:
    vtype = var.vtype()
    _put(row, fm, "obj_coef", var.getObj() / obj_norm)
    _put(row, fm, "binary", float(vtype == "BINARY"))
    _put(row, fm, "integer", float(vtype == "INTEGER"))
    _put(row, fm, "implicit_integer", float(vtype == "IMPLINT"))
    _put(row, fm, "continuous", float(vtype == "CONTINUOUS"))


def _fill_dynamic_var(row, model: Model, var: Variable, col: Column, obj_norm: float, n_lps: float, fm: dict) -> None:
    lb, ub = col.getLb(), col.getUb()
    solval = col.getPrimsol()
    has_lb = not model.isInfinity(abs(lb))
    has_ub = not model.isInfinity(abs(ub))

    _put(row, fm, "has_lb", float(has_lb))
    _put(row, fm, "has_ub", float(has_ub))
    _put(row, fm, "red_cost", model.getVarRedcost(var) / obj_norm)
    _put(row, fm, "sol_val", solval)
    _put(row, fm, "sol_frac", _sol_frac(model, col))
    _put(row, fm, "sol_at_lb", float(has_lb and model.isEQ(solval, lb)))
    _put(row, fm, "sol_at_ub", float(has_ub and model.isEQ(solval, ub)))
    _put(row, fm, "age", col.getAge() / (n_lps + _CSTE))

    if "best_incumbent_val" in fm:
        _put(row, fm, "best_incumbent_val", _incumbent_val(model, var))
    if "avg_incumbent_val" in fm:
        _put(row, fm, "avg_incumbent_val", _avg_incumbent_val(model, var))

    basis = col.getBasisStatus()
    _put(row, fm, "basis_lower", float(basis == "lower"))
    _put(row, fm, "basis_basic", float(basis == "basic"))
    _put(row, fm, "basis_upper", float(basis == "upper"))
    _put(row, fm, "basis_zero", float(basis == "zero"))


# ---------------------------------------------------------------------------
# Per-row feature filling
# ---------------------------------------------------------------------------


def _fill_static_row(
    row, model: Model, scip_row, lhs_or_rhs: float, sign: float, row_norm: float, obj_norm: float, fm: dict
) -> None:
    _put(row, fm, "bias", sign * (lhs_or_rhs - scip_row.getConstant()) / row_norm)
    _put(row, fm, "obj_cosine", sign * _obj_cosine(model, scip_row, obj_norm))


def _fill_dynamic_row(
    row,
    model: Model,
    scip_row,
    lhs_or_rhs: float,
    sign: float,
    row_norm: float,
    obj_norm: float,
    n_lps: float,
    fm: dict,
) -> None:
    _put(row, fm, "is_tight", float(model.isEQ(model.getRowLPActivity(scip_row), lhs_or_rhs)))
    _put(row, fm, "dual_sol", sign * scip_row.getDualsol() / (row_norm * obj_norm))
    _put(row, fm, "age", scip_row.getAge() / (n_lps + _CSTE))


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def _count_ineq_rows(model: Model, rows) -> tuple[int, int]:
    n_ineq = 0
    nnz = 0
    for row in rows:
        row_nnz = row.getNLPNonz()
        if not model.isInfinity(abs(row.getLhs())):
            n_ineq += 1
            nnz += row_nnz
        if not model.isInfinity(abs(row.getRhs())):
            n_ineq += 1
            nnz += row_nnz
    return n_ineq, nnz


def _extract(
    model: Model,
    col_fm: dict,
    row_fm: dict,
    prev_var_features=None,
    prev_row_features=None,
    prev_edge_features=None,
    suppress_warnings: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[NDArray[np.int64], NDArray[np.float64]]]:
    """Returns (var_features, row_features, (edge_indices, edge_vals))."""
    vars_and_cols = _get_vars_and_cols(model)
    rows = model.getLPRowsData()
    obj_norm = _obj_l2_norm([v for v, _ in vars_and_cols])
    n_lps = float(model.getNLPs())
    n_vars = len(vars_and_cols)
    n_ineq, nnz = _count_ineq_rows(model, rows)

    # -- Variable features --
    update_static_var = True
    if prev_var_features is not None:
        if prev_var_features.shape == (n_vars, len(col_fm)):
            var_features = prev_var_features
            update_static_var = False
        else:
            if not suppress_warnings:
                warnings.warn("Variable features shape changed; recomputing static features.", stacklevel=3)
            var_features = np.zeros((n_vars, len(col_fm)), dtype=np.float64)
    else:
        var_features = np.zeros((n_vars, len(col_fm)), dtype=np.float64)

    for i, (var, col) in enumerate(vars_and_cols):
        if update_static_var:
            _fill_static_var(var_features[i], var, obj_norm, col_fm)
        _fill_dynamic_var(var_features[i], model, var, col, obj_norm, n_lps, col_fm)

    # -- Row features --
    update_static_row = True
    if prev_row_features is not None:
        if prev_row_features.shape == (n_ineq, len(row_fm)):
            row_features = prev_row_features
            update_static_row = False
        else:
            if not suppress_warnings:
                warnings.warn("Row features shape changed; recomputing static features.", stacklevel=3)
            row_features = np.zeros((n_ineq, len(row_fm)), dtype=np.float64)
    else:
        row_features = np.zeros((n_ineq, len(row_fm)), dtype=np.float64)

    # -- Edge features --
    update_edges = True
    if prev_edge_features is not None:
        prev_idx, prev_vals = prev_edge_features
        if prev_vals.shape == (nnz,):
            edge_indices, edge_vals = prev_idx, prev_vals
            update_edges = False
        else:
            if not suppress_warnings:
                warnings.warn("Edge features shape changed; recomputing.", stacklevel=3)
            edge_indices = np.zeros((2, nnz), dtype=np.int64)
            edge_vals = np.zeros(nnz, dtype=np.float64)
    else:
        edge_indices = np.zeros((2, nnz), dtype=np.int64)
        edge_vals = np.zeros(nnz, dtype=np.float64)

    feat_row = 0
    edge_j = 0
    for scip_row in rows:
        lhs = scip_row.getLhs()
        rhs = scip_row.getRhs()
        has_lhs = not model.isInfinity(abs(lhs))
        has_rhs = not model.isInfinity(abs(rhs))
        row_norm = scip_row.getNorm()
        if not model.isGT(row_norm, 0.0):
            row_norm = 1.0
        row_cols = scip_row.getCols()
        row_vals = scip_row.getVals()
        row_nnz = scip_row.getNLPNonz()

        for side_val, sign, has_side in ((lhs, -1.0, has_lhs), (rhs, 1.0, has_rhs)):
            if not has_side:
                continue
            if update_static_row:
                _fill_static_row(row_features[feat_row], model, scip_row, side_val, sign, row_norm, obj_norm, row_fm)
            _fill_dynamic_row(
                row_features[feat_row], model, scip_row, side_val, sign, row_norm, obj_norm, n_lps, row_fm
            )
            if update_edges:
                for k in range(row_nnz):
                    edge_indices[0, edge_j] = feat_row
                    edge_indices[1, edge_j] = _edge_col_index(row_cols[k])
                    edge_vals[edge_j] = sign * row_vals[k] / row_norm
                    edge_j += 1
            else:
                edge_j += row_nnz
            feat_row += 1

    return var_features, row_features, (edge_indices, edge_vals)


def _update_dynamic_only(
    model: Model, var_features: np.ndarray, row_features: np.ndarray, col_fm: dict, row_fm: dict
) -> None:
    """Update only dynamic features in-place (used by cache mode)."""
    vars_and_cols = _get_vars_and_cols(model)
    obj_norm = _obj_l2_norm([v for v, _ in vars_and_cols])
    n_lps = float(model.getNLPs())

    for i, (var, col) in enumerate(vars_and_cols):
        _fill_dynamic_var(var_features[i], model, var, col, obj_norm, n_lps, col_fm)

    feat_row = 0
    for scip_row in model.getLPRowsData():
        row_norm = scip_row.getNorm()
        if not model.isGT(row_norm, 0.0):
            row_norm = 1.0
        for side_val, sign, has_side in (
            (scip_row.getLhs(), -1.0, not model.isInfinity(abs(scip_row.getLhs()))),
            (scip_row.getRhs(), 1.0, not model.isInfinity(abs(scip_row.getRhs()))),
        ):
            if not has_side:
                continue
            _fill_dynamic_row(
                row_features[feat_row], model, scip_row, side_val, sign, row_norm, obj_norm, n_lps, row_fm
            )
            feat_row += 1


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class NodeBipartiteEcole:
    """Pure-Python bipartite graph observation with configurable feature extraction.

    Parameters
    ----------
    cache : bool
        When ``True``, static features and edge structure are computed once (at
        the root node or first call) and reused; only dynamic features are
        refreshed on subsequent calls. Mirrors Ecole's ``NodeBipartite(cache=True)``.
    suppress_warnings : bool
        Suppress shape-change warnings when passing ``prev_*`` features.
    static_col_features, dynamic_col_features : tuple[str, ...]
        Column features to extract.
    static_row_features, dynamic_row_features : tuple[str, ...]
        Row features to extract.
    """

    def __init__(
        self,
        cache: bool = False,
        suppress_warnings: bool = False,
        static_col_features: tuple = _DEFAULT_STATIC_COL,
        dynamic_col_features: tuple = _DEFAULT_DYNAMIC_COL,
        static_row_features: tuple = _DEFAULT_STATIC_ROW,
        dynamic_row_features: tuple = _DEFAULT_DYNAMIC_ROW,
    ) -> None:
        self.cache = cache
        self.suppress_warnings = suppress_warnings

        col_feats = tuple(static_col_features) + tuple(dynamic_col_features)
        row_feats = tuple(static_row_features) + tuple(dynamic_row_features)

        self._col_fm = {f: i for i, f in enumerate(col_feats)}
        self._row_fm = {f: i for i, f in enumerate(row_feats)}

        self._cached_var_features = None
        self._cached_row_features = None
        self._cached_edge_features = None
        self._cache_computed = False

    def reset(self, model: Model) -> None:
        self._cached_var_features = None
        self._cached_row_features = None
        self._cached_edge_features = None
        self._cache_computed = False

    def _extract_tuple(
        self, model: Model, done: bool, prev_var_features=None, prev_row_features=None, prev_edge_features=None
    ) -> tuple | None:
        """Extract the bipartite graph observation.

        Returns
        -------
        tuple or None
            ``(variable_features, row_features, (edge_indices, edge_vals))``
            during solving, or ``None`` when done or outside the solving stage.
            ``edge_indices`` has shape ``(2, nnz)``; ``edge_vals`` has shape ``(nnz,)``.
        """
        if done or model.getStage() != PY_SCIP_STAGE.SOLVING:
            return None

        if self.cache:
            current = model.getCurrentNode()
            on_root = current is not None and current.getDepth() == 0
            if on_root or not self._cache_computed:
                vf, rf, ef = _extract(model, self._col_fm, self._row_fm, suppress_warnings=self.suppress_warnings)
                self._cached_var_features = vf
                self._cached_row_features = rf
                self._cached_edge_features = ef
                self._cache_computed = True
                return vf, rf, ef
            assert self._cached_var_features is not None
            assert self._cached_row_features is not None
            vf = self._cached_var_features.copy()
            rf = self._cached_row_features.copy()
            _update_dynamic_only(model, vf, rf, self._col_fm, self._row_fm)
            return vf, rf, self._cached_edge_features

        return _extract(
            model,
            self._col_fm,
            self._row_fm,
            prev_var_features=prev_var_features,
            prev_row_features=prev_row_features,
            prev_edge_features=prev_edge_features,
            suppress_warnings=self.suppress_warnings,
        )

    def extract(
        self, model: Model, done: bool, prev_var_features=None, prev_row_features=None, prev_edge_features=None
    ) -> BipartiteGraph | None:
        """Extract the bipartite graph observation.

        Returns
        -------
        BipartiteGraph or None
            A :class:`BipartiteGraph` dataclass during solving,
            or ``None`` when done or outside the solving stage.
        """
        result = self._extract_tuple(model, done, prev_var_features, prev_row_features, prev_edge_features)
        if result is None:
            return None
        variable_features, row_features, (edge_indices, edge_features) = result
        bg = BipartiteGraph(
            variable_features=variable_features,
            row_features=row_features,
            edge_features=EdgeFeatures(indices=edge_indices, values=edge_features),
        )
        return bg

    @classmethod
    def getBipartiteGraphRepresentation(
        cls,
        model: Model,
        static_only: bool = False,
        static_col_features: tuple | None = None,
        dynamic_col_features: tuple | None = None,
        static_row_features: tuple | None = None,
        dynamic_row_features: tuple | None = None,
        prev_col_features=None,
        prev_row_features=None,
        prev_edge_features=None,
        suppress_warnings: bool = False,
    ) -> tuple:
        """Stateless extraction returning a 4-tuple.

        Raises ``RuntimeError`` when the model is not in SOLVING stage.

        Returns
        -------
        tuple
            ``(col_features, (edge_indices, edge_values), row_features, {})``
            where ``edge_indices`` has shape ``(2, nnz)`` and
            ``edge_values`` has shape ``(nnz,)``.
        """
        if model.getStage() != PY_SCIP_STAGE.SOLVING:
            raise RuntimeError(
                f"Model must be in SOLVING stage to extract observations, got stage {model.getStage()!r}."
            )
        kwargs: dict = dict(suppress_warnings=suppress_warnings)
        if static_only:
            kwargs["dynamic_col_features"] = ()
            kwargs["dynamic_row_features"] = ()
        if static_col_features is not None:
            kwargs["static_col_features"] = static_col_features
        if dynamic_col_features is not None:
            kwargs["dynamic_col_features"] = dynamic_col_features
        if static_row_features is not None:
            kwargs["static_row_features"] = static_row_features
        if dynamic_row_features is not None:
            kwargs["dynamic_row_features"] = dynamic_row_features

        obs = cls(**kwargs)
        result = obs._extract_tuple(
            model,
            done=False,
            prev_var_features=prev_col_features,
            prev_row_features=prev_row_features,
            prev_edge_features=prev_edge_features,
        )
        if result is None:
            raise RuntimeError("Model is not in SOLVING stage.")
        variable_features, row_features, (edge_indices, edge_values) = result
        return (variable_features, (edge_indices, edge_values), row_features, {})
