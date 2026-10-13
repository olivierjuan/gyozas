# Fidelity Profiles

::: gyozas.fidelity

Un profil de fidélité (R5.4) recharge le régime natif d'une méthode — features, expert, augmentation, paramètres solveur — par simple déclaration :

```python
from gyozas import FidelityProfile, Method, Runner, register_fidelity_profile

register_fidelity_profile(
    FidelityProfile(
        name="mymodel",
        observation=lambda: NodeBipartite(),          # ses features
        scip_params={"randomization/permutevars": True},
        training={"loss": "...", "schedule": "..."},  # méta : code d'entraînement, pas exécuté
    )
)

Method(name="mymodel", policy=lambda: MyPolicy(), fidelity="mymodel")
```

Sans politique propre (`policy=None`), l'expert du profil pilote le solveur — avec repli sur la règle par défaut là où il ne décide pas. Les spécificités d'entraînement (lissage 2e choix de Lookback, H3 de TGPPO, pré-entraînement contrastif de Mamba, ...) sont **enregistrées mais non exécutées** : elles appartiennent au code d'entraînement de la méthode, hors périmètre du harnais.

## Les huit régimes documentés

Les régimes de la checklist §3 de la spécification sont enregistrés par défaut (R5.4), d'après les dépôts officiels (voir [References](references.md)) :

| Profil | Régime | Expert natif | Features | Augmentation |
|---|---|---|---|---|
| `gasse` | bipartite-LP | strong branching | `NodeBipartite` (GCNN) | — |
| `lookback` | bipartite-LP | strong branching | `NodeBipartite` (GCNN) | — + loss (lissage 2e choix + PAT) |
| `treegate` | solver-stats | relpscost | état d'arbre 61-dim, gating | grille (s ∈ 0..4, k ∈ 0/1/5/10/15) |
| `tbrant` | solver-stats | relpscost | ~20 solveur + ~40 arbre, GAT + BiMatching | Zarpellon |
| `mamba` | solver-stats | relpscost | embeddings TreeGate + SSM sur le chemin | grille TreeGate |
| `tgppo` | RL (PPO) | aucun | 25 candidat + 8 nœud + 53 arbre | 25 × 5 graines |
| `retrobranching` | RL | aucun | paramétrable (bipartite par défaut) | 5 graines d'épisodes |
| `scavuzzo` | RL (tree MDP) | aucun | crédit d'attribution en arbre | — |

```python
from gyozas import available_fidelity_profiles, get_fidelity_profile

available_fidelity_profiles()          # les huit régimes documentés + les vôtres
get_fidelity_profile("lookback").training   # loss, sélection de modèle, protocole SCIP...
```

Notes : les protocoles SCIP de chaque papier (6.0.1 sauf Mamba 8.0.4) sont enregistrés en métadonnée — le support dual de SCIP est l'exigence R2.1 ; la grille d'augmentation (s, k) est un paramètre de collection, un `ZarpellonAugmentation` par paire ; les extracteurs de features solver-stats (61-dim TreeGate, ~60 T-BranT, 25+8+53 TGPPO) restent à implémenter par méthode, vérifiés contre chaque dépôt officiel.
