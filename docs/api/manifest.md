# Instance Manifest

The instance manifest is the frozen unit of a benchmark protocol (G2):
every instance is hashed, counted and attributed provenance, and the
manifest itself carries a content hash that plays the role of a
signature — after the freeze, any change to any instance is detected.

Instances are addressed by `(strate, split, name)`. The splits are
embedded in the data, never rebuilt by the harness.

::: gyozas.manifest.InstanceManifest

::: gyozas.manifest.file_hash
