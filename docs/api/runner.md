# Experiment Runner

::: gyozas.runner

Le runner (R5.1/R5.2/R5.3/R8.3) exécute une grille d'expérience — méthode × strate × split × graine × profil solveur — décrite par un fichier de configuration JSON :

```python
from gyozas import ExperimentConfig, Method, Runner
from gyozas.policy import CallablePolicy

config = ExperimentConfig.load("experiment.json")   # la grille de runs
methods = {
    "mymodel": Method(
        name="mymodel",
        policy=lambda: MyTrainedPolicy(),           # une fabrique par run
        config={"checkpoint": "model.pt"},
    ),
    "relpscost-baseline": Method(
        name="relpscost-baseline",
        policy=lambda: CallablePolicy(relpscost_action),
    ),
}
runner = Runner(config, methods, {"miplib_s": "data/miplib_s"}, "results/")
runner.run()
```

Propriétés garanties :

- **Idempotence** : un run dont le record JSON existe et est complet n'est jamais rejoué — relancer la même config est un no-op.
- **Reprise** : un record absent ou incomplet (run interrompu) est rejoué au lancement suivant.
- **Garde-fou 5 graines** (`MIN_SEEDS`) : une cellule incomplète ou sous-grainée ne produit pas de résultat publiant (`Runner.is_publishable`, `Runner.collect_cell`).
- **Artefacts naturels** : chaque record embarque son `RunIdentity` (config solveur exacte, triplet de graines, versions SCIP/pyscipopt/gyozas, hash du manifeste d'instances), et la config de l'expérience est archivée avec les résultats — rien n'est assemblé à la main en fin de projet.

Le layout des résultats :

```
results/
├── experiment_config.json
└── <method>/<strate>/<split>/<seed>/<instance>.json
```
