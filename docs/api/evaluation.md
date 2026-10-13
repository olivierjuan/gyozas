# Evaluation

::: gyozas.evaluation

Évaluer une politique en B&B complet contre les baselines classiques — solveur par défaut (= l'unité 1.000×), pscost, relpscost, SB bornée — sur les **mêmes instances et graines** :

```python
from gyozas import RelpscostExpert, evaluate
from gyozas.policy import CallablePolicy

result = evaluate(
    methods={
        "mymodel": my_trained_policy,       # une Policy (R1.1)
        "relpscost": RelpscostExpert(),     # l'expert scalable (R4.1)
        # "default" (le solveur par défaut) est ajouté implicitement comme baseline
    },
    instance_files={"setcover300": "tests/instance.lp"},
    seeds=[0, 1, 2, 3, 4],                  # 5 graines minimum (le garde-fou R5.2 s'applique)
    scip_params={"limits/nodes": 1000},
)

result["sgm"]          # SGM des ratios par instance, baseline = 1.0
result["ratios"]       # ratios appariés par instance (l'unité R6.2)
result["summary"]      # tableau double reporting : SGM, p50/p95, pires instances, résolues vs tronquées
```

Un test de rang apparié sur les nœuds suit ensuite naturellement (`friedman_nemenyi`, `wilcoxon_pair` — R7.3).
