# Run Identity

The complete identity of a run (R8.1): method and exact solver config,
seed triplet, SCIP version, pyscipopt version, harness version, and the
hash of the instance manifest. Two runs with the same identity are the
same run — a reader of a results table can replay a number without
ambiguity.

::: gyozas.run_identity.RunIdentity
