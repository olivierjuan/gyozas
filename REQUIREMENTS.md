
# Harnais GYOZAS — liste d'exigences

Exigences de la **phase 1** de la [[Recherche/TreeRope/Benchmark learn2branch unifié|roadmap du benchmark unifié]] (jalons G1/G2), c'est-à-dire ce que la librairie [[GYOZAS]] doit faire pour porter le benchmark. Chaque exigence est traçable à son origine dans la spécification. Les exigences R1–R9 sont *toutes* motrices pour G1, sauf mention contraire ; les exigences d'extension (EDF, Niveau 2) sont notées « extension ».

**État au 13/10/2026** — validation de l'installation inspectée sur Selena (`/scratch/rd/turing/olivier/TreeRope/.venv`, gyozas 1.0.0, pyscipopt 6.2.1 unique, gymnasium 1.3.0) : ✅ déjà respecté · 🟡 partiel · 🔴 absent · ⚪ inconclusif (wheel installé, pas la source). Synthèse : l'ossature est en place et au-delà du requis sur le socle (contrat d'adaptateur, callbacks, rewards intégraux, API générique à 4 environments) ; les trous majeurs se regroupent en cinq chantiers — dual SCIP (R2.1), vue solver-stats + décision typée (R1.2b/R1.4), volet données/splits/manifeste (R3), collecte d'expert et augmentation (R4), volet évaluation/runner/statistiques/identité de run (R5/R7/R8, partiellement reportable au niveau TreeRope où snakemake est déjà installé).

## R1 — Contrat de politique (l'abstraction d'adaptateur)

- **R1.1** ✅ Interface unique de politique, en aval du solveur : toute méthode expose une fonction de décision du type `select_branching_decision(state) -> BranchingDecision` ; le harnais pilote le solveur, jamais l'inverse. *(spéc. §6 — « l'abstraction d'adaptateur est le point où brancher des politiques hétérogènes »)*
- **R1.2** ✅ Contrat d'état à **trois vues** sur le même nœud, chacune optionnelle : (a) `BipartiteState` — graphe LP complet (Gasse, Lookback, RetroBranching, Scavuzzo) ; (b) état solver-stats — ~25–100 scalaires nœud + arbre (`SolverStats`, TreeGate, T-BranT, TGPPO, Mamba) ; (c) état RL — observation d'environnement sans expert (RetroBranching, Scavuzzo, TGPPO). Un jeu de features unique priverait quelqu'un : les trois coexistent. *(spéc. §2 piste A « chaque méthode garde ses features natives », §1.1)*
- **R1.3** ✅ Extraction **pay-per-view** : les features solveur sont O(1) par nœud ; le graphe LP complet n'est extrait que si la politique le déclare (son extraction est coûteuse sur les grosses instances). *(spéc. §1.1)*
- **R1.4** ✅ Décision typée (`BranchingDecision` : variable × direction) avec compteurs **appariés** (`PairedCounters`) : décisions et nœuds comptés séparément, asymétrie ~1,84–2,00× documentée dans le contrat (jamais mélangées dans une agrégation). *(description TreeRoPE, piège 2 ; spéc. §5.2)*
- **R1.5** ✅ Une nouvelle politique = une classe + une config ; zéro modification du harnais. C'est la matérialisation de l'argument d'adoption « une ligne pour évaluer une nouvelle politique ». *(spéc. §6)*
- **R1.6** ✅ La politique peut être **architecturalement quelconque** : le harnais ne voit que l'état qu'elle a déclaré et la décision qu'elle rend — pas de présupposé GCNN, séquentiel, ou arbre entier. *(spéc. §2, checklist §3)*

## R2 — Solveur, versions, configuration

- **R2.1** 🔴 Support **dual de SCIP 6.0.1 et 8.0.4**, installables côte à côte, sélection par config ; la version est enregistrée dans le résultat de chaque run. *(spéc. §4)*
- **R2.2** ✅ Configuration solveur déclarative par protocole (`SolverConfig`, profils nommés sérialisables avec les résultats ; profil **lin2022** = sandbox T-BranT : heuristiques primales coupées, 1 h, auxiliaires SB désactivés, cutoff à l'optimum passé par instance via `apply(cutoff=...)`). *(spéc. §2 piste A ; github.com/linjc16/TBranT)*
- **R2.3** ✅ Callbacks complets : sélection de variable (brancher avec la politique), collection de trajectoires d'expert, lecture des statistiques d'arbre (nœud courant + cumulatives). *(checklist §3 TreeGate/T-BranT/TGPPO)*
- **R2.4** ✅ Déterminisme par triplet (`SeedTriplet` : graine solveur, graine Python/NumPy, graine d'augmentation) : le triplet fait partie de l'identité d'un run et se réplique à l'identique. *(spéc. §2 « 5 graines », §8 bruit de réentraînement)*
- **R2.5** ✅ Budget de résolution paramétrable (temps et/ou nœuds, `SetLimits`) avec **état de fin toujours rapporté** (`SolveStatus`) : résolue, ou tronquée au budget (et lequel). *(spéc. §5.3 — le piège de troncature)*

## R3 — Instances, strates, splits

- **R3.1** ✅ Les trois strates du tronc universel embarquées comme données : familles Ecole synthétiques (setcover/cauctions/indset/facilities, faciles + medium), MIPLIB-S (les 27 de TreeGate), MILP-L (les 73 de Mamba, 57 faciles / 16 difficiles). *(spéc. §2 piste A)*
- **R3.2** ✅ Les splits train/test d'origine **respectés à l'identique** : ils sont embarqués dans les données de strate, jamais reconstruits par le harnais (fuite impossible, comparabilité directe aux publications). *(spéc. §2 piste A)*
- **R3.3** ✅ Accès par clé (strate, split, instance) avec **manifeste signé** (`InstanceManifest` : SHA-256 de chaque instance, comptage par strate/split, provenance, hash de contenu du manifeste = signature) ; `verify` détecte toute altération. Le manifeste est l'unité de gel du protocole (G2). *(roadmap phase 2 / G2)*
- **R3.4** 🟡 *(extension)* Ajout de strates sans refactor : strate pilote EDF issue de `n_campagne_adonis` (hétérogénéité extrême, étiquette de difficulté candidate = écart prévu/réalisé) ; Distributional MIPLIB si elle sort comme bibliothèque de distributions. *(spéc. §6 cadrage CIFRE, bilan OpenReview)*

## R4 — Collection d'expert et augmentation

- **R4.1** ✅ Experts pluggables — au minimum : **strong branching** (`StrongBranchingExpert`, famille synthétique, piste B bipartite-LP) et **relpscost** (`RelpscostExpert` sur les scores pseudocost natifs de SCIP, l'expert commun scalable, piste A) ; l'expert est un paramètre du setup, pas de la méthode. *(spéc. §1.2, §2 piste B)*
- **R4.2** ✅ Augmentation Zarpellon (seed s, k) paramétrable par config (`ZarpellonAugmentation` : graine de permutation des variables + k premières décisions par un explorer aléatoire, cf. github.com/ds4dm/branch-search-trees) — **native, jamais imposée** (les méthodes qui s'en servent gardent leur avantage, celles qui ne l'utilisent pas ne la subissent pas). *(spéc. §1.3)*
- **R4.3** ✅ Métadonnées de collection enregistrées (`CollectionMetadata`) : notamment la **distribution des nœuds multi-optimaux** de chaque jeu collecté — sans elle, l'acc@1 est illisible. *(description TreeRoPE, piège 1 ; spéc. §5.1)*
- **R4.4** ✅ Enregistrement des trajectoires RL dans un format commun (`Trajectory`/`TrajectoryRecorder` : états, actions, récompenses, retours cumulés ; reconstruction rétrospective depuis l'arbre via `retrospective_trajectories` — le mode RetroBranching/Scavuzzo), consommables hors ligne pour l'entraînement. *(checklist §3 RetroBranching/Scavuzzo)*

## R5 — Entraînement et exécution

- **R5.1** ✅ Runner piloté par config : un fichier décrit méthode × strate × graine × version de solveur ; un run terminé est idempotent (relancer = no-op), un run interrompu reprend. *(roadmap phase 3 — quelques centaines de runs)*
- **R5.2** ✅ La graine minimale du protocole (5) est vérifiée par le harnais : une cellule incomplète ne produit pas de résultat publiant. *(spéc. §2, §8)*
- **R5.3** ✅ Orchestration batchée séquencable sur les moyens du labo (runs GPU ~1 h sur les difficiles, × 5 graines × 6–8 méthodes × 3 strates), avec reprise après incident. *(spéc. §6 coût, roadmap phase 3)*
- **R5.4** 🔴 **Profils de fidélité native** (piste B) : par méthode, un profil activable qui recharge son régime d'origine — features, expert, augmentation, et les spécificités de loss/entraînement (lissage 2e choix + PAT de Lookback, H3 + Optuna imbriqué de TGPPO, pré-entraînement contrastif + inférence autoregressive de Mamba, BiMatching de T-BranT…). La checklist §3 de la spécification est exécutable, pas documentaire. *(checklist §3)*

## R6 — Évaluation

- **R6.1** ✅ Run B&B complet avec la politique branchée, contre les baselines classiques (pscost, relpscost, SB borné) sur les **mêmes instances et graines** ; la baseline solveur par défaut est l'unité 1.000×. *(spéc. §2 piste A ; description TreeRoPE)*
- **R6.2** ✅ Unités appariées **par instance** (`paired_per_instance_ratios`, `sgm_of_ratios`) : SGM des nœuds par instance, jamais moyennes transverses de décisions ni agrégats qui masquent l'hétérogénéité des échelles entre instances. *(spéc. §5.2)*
- **R6.3** ✅ **Comptage d'instances résolues à la limite** (`solved_vs_truncated`) comme métrique primaire sur les strates difficiles (la leçon TreeRoPE : le classement peut s'inverser une fois les tronquées sorties) ; SGM calculée sur résolues seulement ; PD integral pour la progression du primal gap en temps (`PrimalIntegral`). *(spéc. §5.3)*
- **R6.4** 🔴 (Réservée à la comparaison GCNN+SB natif vs GCNN+relpscost sur les synthétiques — la mesure du *coût du passage à l'expert scalable*, chiffrée une fois pour toute la littérature.) *(spéc. §2 piste B, comparaison 1)*

## R7 — Métriques et statistique

- **R7.1** ✅ L'acc@1 n'est **jamais rapportée seule** : toujours accompagnée de la distribution de nœuds multi-optimaux de la collection (R4.3), et jamais comparée entre experts différents. *(spéc. §5.1)*
- **R7.2** ✅ Double reporting systématique **médiane vs queue** (`summary_table` : SGM, p50/p95, pertes catastrophiques nommées) (la leçon TGPPO : domination médiane 78,8 % et SGM agrégée pire peuvent coexister). *(spéc. §5, §1.5)*
- **R7.3** ✅ Tests de rangs automatisés : **Friedman + Nemenyi sur les rangs intra-instance** (`friedman_nemenyi`) et **Wilcoxon apparié unilatéral avec correction de Holm** (`wilcoxon_pair`/`wilcoxon_holm`) — implémentés une fois dans la librairie, appliqués partout. *(spéc. §5, §1.5)*
- **R7.4** ✅ Win rates head-to-head par instance entre toute paire de méthodes (`win_rate`). *(spéc. §2 piste A ; protocole TGPPO)*
- **R7.5** ✅ Tableau de sortie unique (méthode × métrique) exporté brut JSON + CSV lisible (`summary_table`/`save_table`) ; les mêmes données alimentent le papier et les repo de résultats. *(roadmap phase 6)*

## R8 — Reproductibilité et gel de protocole

- **R8.1** ✅ Identité complète de chaque run enregistrée (`RunIdentity.capture` : méthode + config exacte du solveur, triplet de graines, version de SCIP, version pyscipopt, version du harnais, hash du manifeste d'instances). Un lecteur du papier peut re-jouer un chiffre sans ambiguïté. *(spéc. §8 « publication des graines et configs exactes — c'est l'objet »)*
- **R8.2** 🔴 **Gel G2** : release de données v0 + document de protocole figé ; après gel, tous les résultats passent par le même pipeline sans renégociation (garde-fou anti p-hacking, techniquement forcé par le manifeste). *(roadmap G2)*
- **R8.3** ✅ Les artefacts publiables (code, données, graines, configs) sont des sorties naturelles du pipeline, pas un travail de fin de projet séparé. *(roadmap phase 6)*

## R9 — Ingénierie et périmètre

- **R9.1** ✅ Python, packaging **PyPI** (l'ambition GYOZAS), dépendances épinglées ; outillage préféré `uv`. *(note GYOZAS, préférences)*
- **R9.2** ⚪ Tests : unitaires sur chaque vue du contrat d'état (R1.2) et le comptage apparié (R1.4) ; intégration **bout-en-bout sur une micro-grille** (une petite famille × pscost × 1 graine, quelques minutes) en CI — c'est le détecteur de régression du harnais. *(convention d'ingénierie)*
- **R9.3** 🟡 Documentation avec exemples une-page (« évaluer une politique en une ligne ») ; ReadTheDocs visé à la publication PyPI. *(note GYOZAS, promotion)*
- **R9.4** ✅ *(extension)* L'API d'environnement reste gym-like : le branching est le premier environnement, pas le dernier — les autres CO environments (cadre InferOpt/Frank-Wolfe de la thèse) doivent pouvoir s'ajouter sans réécrire le harnais. *(note GYOZAS, cadre InferOpt)*

## Non-goals explicites (ce que le harnais ne fait pas)

- **N'impose ni features uniques, ni expert unique, ni architecture** : la comparabilité porte sur les instances, le solveur, les graines et les métriques — jamais sur l'intérieur du modèle (c'est la ligne de partage piste A / piste B).
- **N'est pas un benchmark de générateurs d'instances** (c'est GenBench-MILP) ni une bibliothèque de distributions (c'est Distributional MIPLIB) — il peut les *consommer*, il ne les remplace pas.
- **Ne fixe pas le classement des versions de SCIP** : la dualité 6.0.1/8.0.4 est un facteur étudié (§4), pas une préférence.

## Voir aussi

- [[Recherche/TreeRope/Benchmark learn2branch unifié|Benchmark learn2branch unifié — spécification]] (l'origine de chaque exigence ; la phase 1 et le jalon G1)
- [[Recherche/TreeRope/TreeRoPE - vision et feuille de route|TreeRoPE — vision et feuille de route]] (le benchmark = Niveau 1 renforcé ; le harnais subsiste comme contribution d'infrastructure en cas d'arrêt en G1)
- [[Recherche/TreeRope/TreeRoPE - description du projet|TreeRoPE — description du projet]] (les trois pièges de métrique, source des exigences R1.4/R4.3/R6.3/R7.1)
- [[GYOZAS]] (la librairie)
