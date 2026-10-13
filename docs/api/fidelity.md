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

Sans politique propre (`policy=None`), l'expert du profil pilote le solveur — avec repli sur la règle par défaut là où il ne décide pas. Les spécificités d'entraînement (lissage 2e choix de Lookback, H3 de TGPPO, pré-entraînement contrastif de Mamba, ...) sont **enregistrées mais non exécutées** : elles appartiennent au code d'entraînement de la méthode, hors périmètre du harnais (scaffolding uniquement — les régimes documentés ne sont pas fournis).
