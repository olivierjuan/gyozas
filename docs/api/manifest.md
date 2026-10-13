# Instance Manifest

The instance manifest is the frozen unit of a benchmark protocol (G2):
every instance is hashed, counted and attributed provenance, and the
manifest itself carries a content hash that plays the role of a
signature — after the freeze, any change to any instance is detected.

Instances are addressed by `(strate, split, name)` — the keys are supplied
by the caller; the splits themselves belong to the strate data and are
never rebuilt by the harness.

::: gyozas.manifest.InstanceManifest

::: gyozas.manifest.file_hash
