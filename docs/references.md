# References

The methods the benchmark covers, with their official implementations (verified October 2026). These are the reference regimes for the [fidelity profiles](api/fidelity.md) (R5.4): a method's native setup — features, expert, augmentation, solver parameters — is recharged from its original code.

## Papers & official repositories

| Paper | Repository | Notes |
|---|---|---|
| Gasse et al. 2019 — *Exact Combinatorial Optimization with Graph Convolutional Neural Networks* | [ds4dm/learn2branch](https://github.com/ds4dm/learn2branch) (+ `learn2branch-ecole`) | 416 ★ |
| Zarpellon et al. 2021 (AAAI) — TreeGate | [ds4dm/branch-search-trees](https://github.com/ds4dm/branch-search-trees) | also the source of the MIPLIB-S strate (27 instances) |
| Lin et al. 2022 — T-BranT | [linjc16/TBranT](https://github.com/linjc16/TBranT) | the `lin2022` solver profile follows its sandbox |
| Gupta et al. 2022 (TMLR) — Lookback | [pg2455/lookback-learn2branch](https://github.com/pg2455/lookback-learn2branch) | Prateek Gupta |
| Parsonson et al. 2022 — RetroBranching | [cwfparsonson/retro_branching](https://github.com/cwfparsonson/retro_branching) | Christopher W. F. Parsonson |
| Scavuzzo et al. 2022 — *Learning to branch with Tree MDPs* (rl2branch) | [lascavana/rl2branch](https://github.com/lascavana/rl2branch) | official implementation, Lara Scavuzzo |
| Benmamed et al. 2025 (AAAI 2026) — TGPPO | [benmhamed-a/tgppo-aaai-26](https://github.com/benmhamed-a/tgppo-aaai-26) | repo published alongside the arXiv deposit |
| Mamba-Branching 2026 | [doctor-watson626/Mamba-Branching](https://github.com/doctor-watson626/Mamba-Branching) | also the source of the MILP-L strate lists (57 easy / 16 hard) |

## Data sources

- **MIPLIB-S strate** — the 27 instances (19 train / 8 test) of the branch-search-trees dataset: [osf.io/uvg5f](https://osf.io/uvg5f/) (Zarpellon et al.).
- **MILP-L strate** — the 73 instances of the Mamba evaluation, sourced from [MIPLIB 2017](https://miplib.zib.de) (`collection.zip` + `benchmark.zip`) and, for the 25 instances absent from those archives, the T-BranT dataset ([Mendeley, doi 10.17632/8msnxmvdgp.1](https://data.mendeley.com/datasets/8msnxmvdgp/1)). See the [benchmark strates](api/strates.md) and the committed manifests' provenance for the exact allocation.
