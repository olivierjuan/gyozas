# Protocol Snapshot

::: gyozas.protocol

Le gel de protocole (R8.2 / jalon G2) : un document unique capturant le contenu du protocole — manifestes de strates (datasets octet-par-octet), profils solveur, versions SCIP/pyscipopt/gyozas — et signé par un hash de contenu :

```python
from gyozas import ProtocolSnapshot

snapshot = ProtocolSnapshot.capture(
    "G2-v0",
    strate_dirs={"miplib_s": "data/miplib_s", "milp_l": "data/milp_l"},
    provenance={"date": "2026-10-13", "decision": "G2 freeze"},
)
snapshot.save("protocol-g2-v0.json")
```

Après gel, `check_run` vérifie qu'une identité de run (R8.1) appartient bien au protocole gelé — dataset figé, builds inchangés — : toute dérive (instance altérée, montée de version, dataset non gelé) est détectée au lieu de passer pour un résultat du protocole. Le gel lui-même reste une *décision* du projet ; le snapshot en est le garde-fou technique (anti p-hacking forcé par le manifeste).
