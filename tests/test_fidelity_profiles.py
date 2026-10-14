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
        assert "masked mean" in profile.training["target"]
        assert "prioritized" in profile.training["replay"]
        assert "10 gradient steps" in profile.training["replay"]
        # DFS node selection with the heuristics left on.
        assert "depth-first" in profile.training["scip_protocol"]
        assert "heuristics ON" in profile.training["scip_protocol"]

    def test_documented_scip_protocols_are_recorded_where_the_paper_has_one(self):
        # The RL papers (RetroBranching, Scavuzzo) leave the SCIP column open in the spec.
        for name in ("gasse", "lookback", "treegate", "tbrant", "mamba", "tgppo", "sorrel"):
            profile = get_fidelity_profile(name)
            assert "6.0.1" in profile.training["scip_protocol"] or "8.0.4" in profile.training["scip_protocol"]
        assert "8.0.3" in get_fidelity_profile("bbmdp").training["scip_protocol"]

    def test_documented_profiles_can_be_reregistered(self):
        assert register_documented_profiles() == DOCUMENTED
