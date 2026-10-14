"""The documented native regimes of the eight benchmark methods (R5.4, spec §3).

Each :class:`~gyozas.fidelity.FidelityProfile` below restates its method's
checklist item from the benchmark specification (§3 « Ce que chaque méthode
exige ») as a harness-activable profile: the parts the harness executes
(observation, expert, solver parameters) and the parts that belong to the
method's own training loop, recorded as metadata (loss, architecture,
pre-training) — the harness does not train.

Notes on what is deliberately *not* hardcoded:

- The solver parameters of each paper's protocol are tied to its SCIP version
  (six of the eight papers run SCIP 6.0.1, Mamba runs 8.0.4). The dual-SCIP
  support is a separate requirement (R2.1); until then the profile records the
  documented protocol in its metadata and the active solver profile stays a
  runner/cell setting (e.g. the ``lin2022`` profile for the universal trunk).
- The augmentation grid of the solver-stats family (s ∈ {0..4} ×
  k ∈ {0, 1, 5, 10, 15}) is a data-collection parameter, one
  :class:`~gyozas.augmentation.ZarpellonAugmentation` per (s, k) — the grid is
  recorded, the concrete (s, k) is chosen by the collection setup.
- The solver-stats features (TreeGate's 61-dim tree state, T-BranT's ~60
  features, TGPPO's 25+8+53) are documented as feature specs; extracting them
  as observation functions is per-method work verified against each official
  repository.
"""

from .experts import RelpscostExpert, StrongBranchingExpert
from .fidelity import FidelityProfile, register_fidelity_profile
from .observations import BranchingStateObservation, NodeBipartite

# Gasse 2019 / Lookback 2022 — the bipartite-LP regime (track B, synthetic Ecole
# families where strong branching is affordable). Lookback is Gasse's regime
# plus its loss: identical features and expert, the difference is in the loss.
_GASSE_TRAINING = {
    "regime": "bipartite-LP",
    "architecture": "GCNN on the bipartite LP graph",
    "instances": "4 Ecole families (setcover/cauctions/indset/facilities), 150k observations, 100 Big instances per family",
    "scip_protocol": "SCIP 6.0.1, root cuts only, no restarts, 45 min limit",
    "native_strate": "ecole synthetics",
}

LOOKBACK_TRAINING = {
    **_GASSE_TRAINING,
    "loss": "second-choice target smoothing + PAT regularizer",
    "model_selection": "practical objectives (Eq. 7-9)",
}

# TreeGate 2021 / T-BranT 2022 / Mamba 2026 — the solver-stats regime (track A
# and B on MIPLIB strates); relpscost is their common expert, the Zarpellon
# augmentation grid is theirs.
_AUGMENTATION_GRID = {"s": list(range(5)), "k": [0, 1, 5, 10, 15]}


def register_documented_profiles() -> list[str]:
    """Register the eight documented fidelity profiles; returns their names."""
    profiles = [
        FidelityProfile(
            name="gasse",
            observation=lambda: NodeBipartite(),
            expert=lambda: StrongBranchingExpert(),
            training=dict(_GASSE_TRAINING),
        ),
        FidelityProfile(
            name="lookback",
            observation=lambda: NodeBipartite(),
            expert=lambda: StrongBranchingExpert(),
            training=dict(LOOKBACK_TRAINING),
        ),
        FidelityProfile(
            name="treegate",
            expert=lambda: RelpscostExpert(),
            training={
                "regime": "solver-stats",
                "features": "61-dim tree state, multiplicative feature gating",
                "augmentation_grid": dict(_AUGMENTATION_GRID),
                "scip_protocol": "SCIP 6.0.1",
                "native_strate": "miplib_s",
            },
        ),
        FidelityProfile(
            name="tbrant",
            observation=lambda: BranchingStateObservation(),
            expert=lambda: RelpscostExpert(),
            training={
                "regime": "solver-stats",
                "features": "25 candidate + 8 node + 53 tree (the BranchingState triple; unavailable slots zeroed on stock PySCIPOpt)",
                "architecture": "full tree as GAT input, transformer without positional encoding, BiMatching",
                "augmentation_grid": {"s": 0, "k": 0, "note": "Zarpellon, 82 407 train / 36 554 test observations"},
                "scip_protocol": "SCIP 6.0.1",
                "native_strate": "miplib 3/2010/2017 + CORAL",
            },
        ),
        FidelityProfile(
            name="mamba",
            expert=lambda: RelpscostExpert(),
            training={
                "regime": "solver-stats",
                "pretraining": "contrastive on TreeGate embeddings",
                "architecture": "SSM (Mamba) over the sequentialized path",
                "inference": "autoregressive — sequential reading of the path is the native mode",
                "augmentation_grid": dict(_AUGMENTATION_GRID),
                "scip_protocol": "SCIP 8.0.4",
                "native_strate": "milp_l (and MILP-S)",
            },
        ),
        FidelityProfile(
            name="tgppo",
            training={
                "regime": "RL (PPO)",
                "features": "25 candidate + 8 node + 53 tree (Zarpellon), transformer over the current node's candidates",
                "reward": "H3, difficulty-adaptive",
                "hpo": "nested Optuna cross-validation (TPE + median pruning)",
                "critic": "mean-aggregated over candidates",
                "augmentation": "25 train x 5 SCIP seeds = 125 episodes, 500 B&B runs",
                "scip_protocol": "SCIP 6.0.1",
            },
        ),
        FidelityProfile(
            name="retrobranching",
            training={
                "regime": "RL",
                "trajectories": "retrospective, reconstructed from the tree (harness: retrospective_trajectories)",
                "credit": "associated return over the reconstructed trajectory",
                "features": "parametrizable (bipartite by default)",
            },
        ),
        FidelityProfile(
            name="scavuzzo",
            training={
                "regime": "RL (tree MDP)",
                "mdp": "tree MDP with its tree policy gradient",
                "features": "tree credit attribution, no history in the input",
            },
        ),
    ]
    for profile in profiles:
        register_fidelity_profile(profile)
    return [p.name for p in profiles]
