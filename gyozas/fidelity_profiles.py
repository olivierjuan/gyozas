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
- The solver-stats family (TreeGate, T-BranT, TGPPO) shares one feature
  parameterization — the 25+8+53 BranchingState triple, Zarpellon's original
  (T-BranT and TGPPO reused it verbatim). All three profiles recharge the
  same :class:`~gyozas.observations.BranchingStateObservation`, whose slots
  unavailable on stock PySCIPOpt are exact zeros, documented per slot.
"""

from .experts import RelpscostExpert, StrongBranchingExpert
from .fidelity import FidelityProfile, register_fidelity_profile
from .observations import BranchingStateObservation, NodeBipartite, NodeBipartiteRetro

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
            observation=lambda: BranchingStateObservation(),
            expert=lambda: RelpscostExpert(),
            training={
                "regime": "solver-stats",
                "features": "25 candidate + 8 node + 53 tree (the BranchingState triple, Zarpellon's original parameterization; unavailable slots zeroed on stock PySCIPOpt)",
                "architecture": "NoTree / TreeGate feedforward, sigmoid feature gating from the tree state",
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
            observation=lambda: BranchingStateObservation(),
            training={
                "regime": "RL (PPO) from scratch — no imitation pretraining",
                "features": "25 candidate + 8 node + 53 tree (the BranchingState triple; unavailable slots zeroed on stock PySCIPOpt)",
                "architecture": "actor-critic transformer: LayerNorm+Linear embeddings (candidates 25->H, node||tree 61->H), candidate-tree concat -> global embedding, TransformerEncoder (GELU, feedforward 4H), BiMatchingNet co-attention refinement, TreeGateBranchingNet gated head (depth 2, layer norm, infimum 1, reduce factor 2) for both the actor's per-candidate scores and the critic's state value (mean-pooled over valid candidates)",
                "reward": "H3, difficulty-adaptive (harness: DifficultyAdaptive): per-step node efficiency, pace vs a power-law expected curve, gap, PDI and progress, weights scheduled by baseline difficulty; terminal bonus by final status (optimal 1+3*speedup, infeasible/unbounded 0.8+2*speedup, timelimit 0.5*speedup+0.3*gap_gain+0.2*pdi_gain, speedup capped 5)",
                "baseline": "one SCIP solve per training instance without the learned brancher, providing baseline_nodes/gap/PDI/status for the reward and the cutoff (objective limit at the optimum)",
                "exploration": "actions sampled from the actor's softmax over the node's LP branching candidates (also at evaluation, with per-job fixed seeds)",
                "optimization": "PPO: clipped surrogate (clip ~0.2), GAE, entropy bonus, AdamW (weight decay 1e-2), gradient-norm clip 0.5, n_epochs over shuffled padded minibatches",
                "hpo": "nested Optuna cross-validation (TPE, 5 outer folds, 60 inner trials): actor/critic lr, hidden dim, layers, heads, dropout, clip, entropy weight, gamma, GAE lambda, batch size, n_epochs, reward in {H1, H2, H3}",
                "augmentation": "25 train instances x 5 SCIP seeds (randomization/permutationseed) = 125 rollout episodes per iteration, 500 B&B runs per score",
                "instances": "MILP benchmark sets (Mendeley) in the paper; this benchmark's protocol families",
                "scip_protocol": "SCIP 6.0.1 (paper), the 'sandbox' setting: all 54 primal heuristics off, cutoff at the known optimum, conflict/usesb off, no probing bounds, feastol/dualfeastol 1e-6, variable permutation seeded per episode; 600 s (train) / 3600 s (test) limit",
            },
        ),
        FidelityProfile(
            name="retrobranching",
            observation=lambda: NodeBipartiteRetro(),
            training={
                "regime": "RL (DQN)",
                "reward": "retro_binary_fathomed (harness: RetroBinaryFathomed): binary sparse scores, retrospective sub-tree episodes (max_leaf_lp_gain, min_subtree_depth=1, fathomed transitions force-included)",
                "features": "43 variable features (19 bipartite-LP + 24 retro), 5 constraint features, edge features on",
                "architecture": "BipartiteGCN: emb 64, 1 round, aggregator add, inverse_leaky_relu activation, head depth 2, normal(0.01) weight init",
                "instances": "set covering 165x230, seed 0",
                "scip_protocol": "SCIP 7.0.5 (ecole 0.8.1), gasse_2019 params (training-side: no cuts, no restarts — a restart renumbers the tree and orphans the sub-tree episodes), 3600 s limit",
                "replay": "uniform, capacity 100k transitions, 20k filled before learning, 1 gradient step per 10 transitions added",
                "optimization": "Adam lr 5e-5, batch 128, MSE, gradient value clipping 10",
                "exploration": "epsilon 0.025, uniform random actions",
                "target": "1-step TD; the online net selects the next action, the target net evaluates it (Double-DQN math); soft Polyak update tau=1e-4 after every gradient step",
                "dead_config": "n_step_return=3 stored in transitions but unused by the loss; use_double_dqn flag off (the non-double branch already computes Double-DQN targets); hard target update every 10k steps short-circuited by the soft update",
            },
        ),
        FidelityProfile(
            name="scavuzzo",
            observation=lambda: NodeBipartite(),
            training={
                "regime": "RL (REINFORCE) fine-tuning over an IL-pretrained policy",
                "pretraining": "exactly the gasse_2019 regime (explore-then-strong-branching, expert_probability 0.05, 100k train / 20k valid samples, 1000 epochs, batch 32, Adam lr 1e-3, entropy bonus 0)",
                "features": "the bipartite-LP observation (Gasse's 5 constraint / 1 edge / 19 variable features); FocusNode is read by the official agent for node bookkeeping only, never as input",
                "architecture": "Gasse's GCNN verbatim (PreNormLayer, emb 64), reused from the gasse profile",
                "instances": "set covering 400x750 (density 0.05) training / 500x600x0.10 in this benchmark's protocol",
                "variants": {
                    "mdp": "flat MDP: return of a decision = minus the number of nodes processed after it (cumulative NNodes at the decision minus at the end)",
                    "tmdp+DFS": "tree MDP with depth-first node selection forced during training (nodeselection/dfs stdpriority = memsavepriority = 666666); return = -(subtree_size) - 1",
                    "tmdp+ObjLim": "tree MDP with setObjlimit(optimal + 0.1) on training instances (eps = -0.1 for maximization); return = -(subtree_size) - 1",
                },
                "reward": "harness: NNodes (mdp) / SubtreeSizes (both tmdp variants); the tree credit needs the objlimit or DFS regime above to be well-defined",
                "exploration": "actions sampled from Categorical(logits) during training, greedy at evaluation; sample_rate 0.05 of transitions kept per episode; episodes beyond 50000 iterations discarded",
                "optimization": "REINFORCE with entropy bonus 1e-5, batches of 16 with gradient accumulation, ONE Adam step (lr 1e-6) per epoch of 10 episodes, instances sampled with replacement",
                "validation": "every 50 epochs: 20 instances x 5 seeds greedy, gmean(nnodes+1)-1, best checkpoint kept; 20000 epochs officially (>= 6 days)",
                "scip_protocol": "SCIP 6.0.1 (paper) / gasse_2019 params + timing/clocktype 2 (training) or 1 (evaluation), 3600 s limit; no restarts — a restart renumbers the tree and orphans the credit",
            },
        ),
    ]
    for profile in profiles:
        register_fidelity_profile(profile)
    return [p.name for p in profiles]
