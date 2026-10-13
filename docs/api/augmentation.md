# Search-Tree Augmentation

The Zarpellon search-tree augmentation (seed s, k): the first k branching
decisions of an episode are taken by a random explorer instead of the
policy, and the solver's variable permutation is seeded with s. Each
(s, k) pair yields a different search-tree prefix and therefore different
states for the policy to be collected on.

The augmentation is **opt-in** — a parameter of the data-collection setup,
never imposed: methods that do not use it are unaffected, methods that do
keep their advantage.

References: Zarpellon et al., "Parameterizing branch-and-bound search trees
to learn branching policies" (2021); github.com/ds4dm/branch-search-trees.

::: gyozas.augmentation.ZarpellonAugmentation
