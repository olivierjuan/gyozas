"""Tests for the documented fidelity profiles (R5.4, spec checklist §3)."""

from gyozas.experts import RelpscostExpert, StrongBranchingExpert
from gyozas.fidelity import available_fidelity_profiles, get_fidelity_profile
from gyozas.fidelity_profiles import register_documented_profiles
from gyozas.observations import NodeBipartite

DOCUMENTED = [
    "gasse",
    "lookback",
    "treegate",
    "tbrant",
    "mamba",
    "tgppo",
    "retrobranching",
    "scavuzzo",
    "sorrel",
    "bbmdp",
    "cambranch",
    "dsccl",
    "fmsts",
]


class TestDocumentedProfiles:
    def test_all_eight_documented_profiles_are_registered(self):
        assert set(DOCUMENTED) <= set(available_fidelity_profiles())

    def test_bipartite_regime_features_and_expert(self):
        for name in ("gasse", "lookback"):
            profile = get_fidelity_profile(name)
            assert isinstance(profile.observation(), NodeBipartite)
            assert isinstance(profile.expert(), StrongBranchingExpert)
            assert profile.training["regime"] == "bipartite-LP"
            assert profile.training["native_strate"] == "ecole synthetics"

    def test_lookback_is_gasse_plus_its_loss(self):
        lookback = get_fidelity_profile("lookback")
        assert lookback.training["loss"] == "second-choice target smoothing + PAT regularizer"
        assert lookback.training["model_selection"]
        # Same regime otherwise.
        assert lookback.observation is not None and lookback.expert is not None

    def test_solver_stats_regime_expert_is_relpscost(self):
        for name in ("treegate", "tbrant", "mamba"):
            profile = get_fidelity_profile(name)
            assert isinstance(profile.expert(), RelpscostExpert)
            assert profile.training["augmentation_grid"] or name == "tbrant"

    def test_tbrant_declares_its_native_observation(self):
        # The runner builds the policy's environment from the profile's
        # observation; a profile without one hands the policy None features
        # (the regression that killed the first tbrant smoke run).
        from gyozas.observations import BranchingStateObservation

        assert isinstance(get_fidelity_profile("tbrant").observation(), BranchingStateObservation)

    def test_treegate_declares_its_native_observation(self):
        # TreeGate parameterized the 25+8+53 triple first (T-BranT and TGPPO
        # reused it); its profile recharges the same observation.
        from gyozas.observations import BranchingStateObservation

        assert isinstance(get_fidelity_profile("treegate").observation(), BranchingStateObservation)

    def test_treegate_augmentation_grid_matches_the_paper(self):
        grid = get_fidelity_profile("treegate").training["augmentation_grid"]
        assert grid == {"s": [0, 1, 2, 3, 4], "k": [0, 1, 5, 10, 15]}

    def test_rl_regimes_have_no_expert(self):
        for name in ("tgppo", "retrobranching", "scavuzzo"):
            profile = get_fidelity_profile(name)
            assert profile.expert is None
            assert profile.training["regime"].startswith("RL")

    def test_retrobranching_declares_its_native_observation(self):
        # The 43-variable-feature observation (19 bipartite + 24 retro). The
        # gasse_2019 parameters stay training-side (the runner would apply
        # profile scip_params at evaluation, breaking comparability — the
        # active solver profile stays a runner/cell setting, like every
        # other profile here).
        from gyozas.observations import NodeBipartiteRetro

        profile = get_fidelity_profile("retrobranching")
        assert isinstance(profile.observation(), NodeBipartiteRetro)
        assert profile.scip_params == {}
        assert profile.training["reward"].startswith("retro_binary_fathomed")
        assert "no restarts" in profile.training["scip_protocol"]

    def test_scavuzzo_declares_its_three_variants(self):
        # The paper offers mdp / tmdp+DFS / tmdp+ObjLim; all three are ported.
        # The observation is Gasse's plain NodeBipartite (the official
        # FocusNode is node bookkeeping, never model input), and the IL
        # pretraining stage is metadata — the harness does not train.
        profile = get_fidelity_profile("scavuzzo")
        assert isinstance(profile.observation(), NodeBipartite)
        assert profile.scip_params == {}
        variants = profile.training["variants"]
        assert set(variants) == {"mdp", "tmdp+DFS", "tmdp+ObjLim"}
        # Both tmdp variants share the tree credit and its harness reward.
        for variant in ("tmdp+DFS", "tmdp+ObjLim"):
            assert "-(subtree_size) - 1" in variants[variant]
        assert "SubtreeSizes" in profile.training["reward"]
        assert "NNodes" in profile.training["reward"]
        assert "no restarts" in profile.training["scip_protocol"]

    def test_mamba_declares_its_two_stage_il_regime(self):
        # The solver-stats family's sequence model: relpscost-expert
        # trajectories, contrastively pretrained TreeGateNet embeddings, then
        # a selective SSM over the sequentialized path. Like every other
        # profile, the sandbox protocol stays training-side metadata.
        from gyozas.observations import BranchingStateObservation

        profile = get_fidelity_profile("mamba")
        assert isinstance(profile.observation(), BranchingStateObservation)
        assert profile.scip_params == {}
        assert profile.training["regime"].startswith("solver-stats IL")
        # The expert is relpscost along trajectories, not strong branching.
        assert "relpscost" in profile.training["regime"]
        assert "not strong branching" in profile.training["regime"]
        # Two stages: contrastive pretraining, then the Mamba sequence model.
        assert "contrastive" in profile.training["pretraining"]
        assert "GroupedMamba" in profile.training["architecture"]
        assert "selective SSM" in profile.training["architecture"]
        assert "deterministic" in profile.training["inference"]
        assert "sandbox" in profile.training["scip_protocol"]
        assert "heuristics off" in profile.training["scip_protocol"]

    def test_tgppo_declares_its_ppo_regime(self):
        # PPO from scratch on the BranchingState triple, H3 reward, the
        # 'sandbox' SCIP setting with per-instance baselines. The sandbox
        # parameters stay training-side like every other RL profile (the
        # active solver profile is a runner/cell setting).
        from gyozas.observations import BranchingStateObservation

        profile = get_fidelity_profile("tgppo")
        assert isinstance(profile.observation(), BranchingStateObservation)
        assert profile.scip_params == {}
        assert profile.training["regime"].startswith("RL (PPO)")
        assert "DifficultyAdaptive" in profile.training["reward"]
        assert "no imitation pretraining" in profile.training["regime"]
        # The baseline solve feeds both the reward and the cutoff.
        assert "cutoff" in profile.training["baseline"]
        assert "sandbox" in profile.training["scip_protocol"]
        assert "heuristics off" in profile.training["scip_protocol"]
        assert "BiMatchingNet" in profile.training["architecture"]
        assert "mean-pooled" in profile.training["architecture"]

    def test_sorrel_declares_its_offline_then_online_regime(self):
        # SORREL 2024 (Feng et al., AAAI 2025): offline RL on suboptimal
        # demonstrations (TD3BC), then online PPO + self-imitation. The
        # demonstrator is VHB — strong branching w.p. 0.05, else pseudocosts —
        # over SCIP's pseudo branching candidates; the tree-MDP machinery and
        # the sandbox protocol stay training-side like every other RL profile.
        profile = get_fidelity_profile("sorrel")
        assert isinstance(profile.observation(), NodeBipartite)
        assert profile.scip_params == {}
        assert profile.training["regime"].startswith("RL (offline then online)")
        assert "no strong-branching expert" in profile.training["regime"]
        # The demonstrator and the pseudo-candidate action space.
        assert "pseudocost" in profile.training["demonstrator"]
        assert "pseudo_candidates=True" in profile.training["action_space"]
        # Both stages, with their harness-visible hooks.
        assert "TD3BC" in profile.training["offline_stage"]
        assert "kappa" in profile.training["offline_stage"]
        assert "Self-Imitation" in profile.training["online_stage"]
        assert "heuristics off" in profile.training["scip_protocol"]
        assert "restarts off" in profile.training["scip_protocol"]

    def test_bbmdp_declares_its_corrected_vanilla_mdp_regime(self):
        # BBMDP 2025 (Forel et al., NeurIPS 2025): the Tree MDP corrected into
        # a vanilla MDP over the visited nodes (Prop. 3.1) — k-step returns
        # recomputed retrospectively, DQN with an HL-Gauss value head, DFS node
        # selection, heuristics ON. The 43-feature observation is recharged
        # from the retrobranching profile; the protocol stays training-side.
        from gyozas.observations import NodeBipartiteRetro

        profile = get_fidelity_profile("bbmdp")
        assert isinstance(profile.observation(), NodeBipartiteRetro)
        assert profile.scip_params == {}
        assert profile.training["regime"].startswith("RL (DQN)")
        assert "no pretraining" in profile.training["regime"]
        assert "corrected vanilla MDP" in profile.training["regime"]
        # The HL-Gauss categorical value head over log2(-z).
        assert "HL-Gauss" in profile.training["value_head"]
        assert "18 bins" in profile.training["value_head"]
        assert "-2^E[centers]" in profile.training["value_head"]
        # The k-step retrospective reward and the per-transition done.
        assert "n_step 3, gamma 1" in profile.training["reward"]
        assert "done = no visited child" in profile.training["next_states"]
        # DDQN multi-children targets and the prioritized replay.
        assert "masked sum" in profile.training["target"]
        assert "prioritized" in profile.training["replay"]
        assert "10 gradient steps" in profile.training["replay"]
        # DFS node selection with the heuristics left on.
        assert "depth-first" in profile.training["scip_protocol"]
        assert "heuristics ON" in profile.training["scip_protocol"]

    def test_cambranch_declares_its_shift_augmented_contrastive_regime(self):
        # CAMBranch 2024 (Lin et al., ICLR 2024 poster): Gasse's IL regime
        # with variable-shifted AMILP augmentations (the FSB labels preserved
        # by Thm 3.1, the augmented features transformed analytically) and a
        # joint contrastive loss. Paper-only port — no official code.
        profile = get_fidelity_profile("cambranch")
        assert isinstance(profile.observation(), NodeBipartite)
        assert profile.scip_params == {}
        assert profile.training["regime"].startswith("IL (FSB)")
        assert "no official repository" in profile.training["note"]
        # The augmentation: free AMILPs by feature transformation, 9 per
        # sample, the shift distribution a declared pilot choice.
        assert "Thm 3.1" in profile.training["augmentation"]
        assert "no re-solve" in profile.training["augmentation"]
        assert "9 AMILPs per original sample" in profile.training["augmentation"]
        assert "not specified in the paper" in profile.training["shift_distribution"]
        # The joint loss and its weights.
        assert "0.05 L_InfoNCE" in profile.training["loss"]
        assert "0.01 L_aux" in profile.training["loss"]
        # Gasse's GCNN plus the projection head (sizes a pilot choice).
        assert "Gasse's GCNN verbatim" in profile.training["architecture"]
        assert "not specified in the paper" in profile.training["projection_head"]
        # Root cuts only — the gasse collection settings coincide.
        assert "root only" in profile.training["scip_protocol"]
        assert "coincide" in profile.training["scip_protocol"]

    def test_dsccl_declares_its_stratified_contrastive_regime(self):
        # DSC-CL / SC-MILP 2025 (Lu, Ma & Tao, arXiv 2511.21107): Gasse's IL
        # regime with dynamic stratified contrastive learning over K-means
        # node groups, upstream-augmented derivations, and hybrid branching
        # (the expert in the inference loop at upstream nodes). Preprint —
        # no official code.
        profile = get_fidelity_profile("dsccl")
        from gyozas.observations import MetaObservation, StrongBranchingScores

        observation = profile.observation()
        assert isinstance(observation, MetaObservation)
        assert isinstance(profile.expert(), StrongBranchingExpert)
        assert isinstance(observation.observations["bipartite"], NodeBipartite)
        assert isinstance(observation.observations["sb_scores"], StrongBranchingScores)
        assert profile.scip_params == {}
        assert profile.training["regime"] == "IL (FSB) + dynamic stratified contrastive learning"
        assert "no official repository" in profile.training["note"]
        # K-means strata: m = 4-5 clusters by elbow.
        assert "K-means" in profile.training["grouping"]
        assert "by elbow" in profile.training["grouping"]
        # The derivations: LT/RC-MILP at 35%, perturbations at 10%, labels kept.
        assert "35% probability" in profile.training["augmentation"]
        assert "10%" in profile.training["augmentation"]
        assert "identical FSB" in profile.training["augmentation"]
        assert "identical LP relaxation" in profile.training["augmentation"]
        # Eq 5: learnable stratified weights, tau = 0.08.
        assert "alpha_0 = 1" in profile.training["loss"]
        assert "tau = 0.08" in profile.training["loss"]
        assert "set to 0 at inference" in profile.training["lambda"]
        # The hybrid: top-5 + FSB upstream (rho = 0.8), direct downstream.
        assert "rho = 0.8" in profile.training["hybrid_branching"]
        assert "top-k candidates (k = 5)" in profile.training["hybrid_branching"]
        # SCIP 7.0.1, 1 h limit.
        assert "SCIP 7.0.1" in profile.training["scip_protocol"]

    def test_fmsts_declares_its_subtree_size_rl_regime(self):
        # FMSTS 2020 (Etheve, Alès, Bissuel, Juan & Kedad-Sidhoum, arXiv
        # 2005.10026): RL from scratch with observable Q-values — the MDP is
        # deterministic, so the TD loss is the plain MSE against the observed
        # subtree sizes. The value function counts nodes; the architecture is
        # multiplicative-dueling; the observation is bespoke (a fitted PCA).
        # No official code.
        profile = get_fidelity_profile("fmsts")
        assert profile.observation is None
        assert profile.expert is None
        assert profile.scip_params == {}
        assert profile.training["regime"].startswith("RL (approximate Q-learning) from scratch")
        assert "no official repository" in profile.training["note"]
        # The subtree-size value function and its optimality under DFS.
        assert "subtree size" in profile.training["value_function"]
        assert "Prop 2" in profile.training["value_function"]
        assert "Prop 1" in profile.training["value_function"]
        assert "Q(s, a) = 1 + V(child0) + V(child1)" in profile.training["value_function"]
        # Eq 5: the MSE weighted so every instance weighs equally.
        assert "weighted by 1/V(tree)" in profile.training["loss"]
        # Eq 6: PER on the error normalised by its target.
        assert "|Q - Q_hat|/Q (Eq 6)" in profile.training["replay"]
        assert "epsilon-greedy" in profile.training["exploration"]
        # The bespoke observation: PCA statics, depth/distance/one-hot dynamics.
        assert "bespoke" in profile.training["observation"]
        assert "PCA of the concatenated (A, b, c)" in profile.training["observation"]
        assert "3|J| one-hot branching state" in profile.training["observation"]
        assert "fitted PCA is part of the checkpoint" in profile.training["observation"]
        # A = J in the paper; the harness offers the LP candidates.
        assert "the harness offers the LP branching candidates" in profile.training["action_space"]
        # MDA: the multiplicative dueling architecture.
        assert "MDA (Multiplicative Dueling Architecture)" in profile.training["architecture"]
        assert "linear-activation" in profile.training["architecture"]
        # One update per iteration (Algorithm 1); the rest are pilot choices.
        assert "one loss-(Eq 5) update per iteration" in profile.training["optimization"]
        assert "not specified" in profile.training["optimization"]
        # Two EDF industrial problems, 200 train / 500 test.
        assert "P1 microgrid" in profile.training["instances"]
        assert "P2 hydroelectric valley" in profile.training["instances"]
        assert "200 training / 500 test" in profile.training["instances"]
        assert profile.training["native_strate"] == "EDF industrial"
        # CPLEX under DFS, everything off — training-side only.
        assert "CPLEX 12.7.1 under DFS" in profile.training["scip_protocol"]

    def test_documented_scip_protocols_are_recorded_where_the_paper_has_one(self):
        # The RL papers (RetroBranching, Scavuzzo) leave the SCIP column open in the spec.
        for name in ("gasse", "lookback", "treegate", "tbrant", "mamba", "tgppo", "sorrel"):
            profile = get_fidelity_profile(name)
            assert "6.0.1" in profile.training["scip_protocol"] or "8.0.4" in profile.training["scip_protocol"]
        assert "8.0.3" in get_fidelity_profile("bbmdp").training["scip_protocol"]
        assert "SCIP 7.0.1" in get_fidelity_profile("dsccl").training["scip_protocol"]
        assert "CPLEX 12.7.1" in get_fidelity_profile("fmsts").training["scip_protocol"]

    def test_documented_profiles_can_be_reregistered(self):
        assert register_documented_profiles() == DOCUMENTED
