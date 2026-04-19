# Instance Modifiers

Modifiers wrap any instance generator and post-process the model before it reaches the environment.

## EmbedObjective

::: gyozas.instances.modifiers.embed_objective.EmbedObjective

## SetParameters

::: gyozas.instances.modifiers.set_parameters.SetParameters

## Convenience Presets

Pre-built `SetParameters` instances for common configurations:

| Name | Effect |
|------|--------|
| `SetNoCuts` | Disables all cutting plane separators |
| `SetNoHeuristics` | Disables all primal heuristics |
| `SetNoDisplay` | Silences SCIP solver output |
| `SetDFSNodeSelection` | Forces depth-first node selection |
| `SetBFSNodeSelection` | Forces breadth-first node selection |
