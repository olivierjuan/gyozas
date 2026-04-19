# Gyozas

<p align="center">
  <img src="https://olivierjuan/gyozas/assets/gyozas.png" alt="Gyozas logo" width="200"/>
</p>

[![CI](https://github.com/olivierjuan/gyozas/actions/workflows/ci.yml/badge.svg)](https://github.com/olivierjuan/gyozas/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gyozas)](https://pypi.org/project/gyozas/)
[![Python](https://img.shields.io/pypi/pyversions/gyozas)](https://pypi.org/project/gyozas/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![codecov](https://codecov.io/gh/olivierjuan/gyozas/branch/main/graph/badge.svg)](https://codecov.io/gh/olivierjuan/gyozas)

A pure-Python reinforcement learning library for combinatorial optimization, built on top of [PySCIPOpt](https://github.com/scipopt/PySCIPOpt) and [Gymnasium](https://gymnasium.farama.org/).

Gyozas provides a modular RL environment that lets you train agents to make decisions inside SCIP's branch-and-bound solver -- variable selection (branching) and node selection -- with pluggable rewards, observations, and problem instance generators.

## Features

- **Gymnasium-style API** -- `reset()` / `step()` / `render()` interface
- **Branching & node selection dynamics** -- control variable branching or node selection decisions
- **Bipartite graph observations** -- LP-based features following [Gasse et al. (NeurIPS 2019)](https://arxiv.org/abs/1906.01629)
- **Multiple reward functions** -- node count, solving time, LP iterations, primal/dual integrals
- **Built-in instance generators** -- set cover, independent set, combinatorial auction, capacitated facility location
- **Branch-and-bound tree visualization**

## Prerequisites

Gyozas requires the [SCIP](https://www.scipopt.org/) solver (version 8+). But latest version should be installed when [PySCIPOpt](https://pyscipopt.readthedocs.io/en/latest/) is installed automatically with Gyozas dependencies

## Installation

```bash
pip install gyozas
```

Or install from source:

```bash
pip install .
```

You can install extra packages for enhanced visualization of branch and bound tree state using

```bash
pip install "gyozas[viz]"
```

## Quick Start

```python
import gyozas

# Create an instance generator
instances = gyozas.SetCoverGenerator(n_rows=100, n_cols=200)

# Create the RL environment
env = gyozas.Environment(
    instance_generator=instances,
    observation_function=gyozas.NodeBipartite(),
    reward_function=gyozas.NNodes(),
)

# Run one episode
obs, action_set, reward, done, info = env.reset()
while not done:
    action = action_set[0]  # pick first available action
    obs, action_set, reward, done, info = env.step(action)

env.close()
```

## Coming from Ecole?

Gyozas is a drop-in replacement for [Ecole](https://github.com/ds4dm/ecole) with a near-identical API:

| | Ecole | Gyozas |
|---|---|---|
| **Install** | C++ build, CMake, custom SCIP | `pip install gyozas` |
| **Language** | C++ core + Python bindings | Pure Python |
| **Maintained** | Last release 2023 | Active |
| **Python** | 3.7--3.10 | 3.10+ |
| **Custom components** | Subclass C++ base classes | Any object with `reset()`/`extract()` |
| **Node selection** | Not supported | `NodeSelectionDynamics` |
| **Gymnasium wrapper** | Not included | `GymnasiumWrapper` for SB3/CleanRL |

See the full [migration guide](https://olivierjuan.github.io/gyozas/migrating-from-ecole/) for detailed API mapping.

## Components

### Dynamics
- `BranchingDynamics` -- control variable selection in branching (default)
- `NodeSelectionDynamics` -- control which node to explore next
- `ConfiguringDynamics` -- single-step algorithm configuration (set SCIP params, then solve)
- `PrimalSearchDynamics` -- LP-probing primal heuristic; agent provides partial variable assignments
- `ExtraBranchingActions` -- sentinel actions (`SKIP`, `CUT_OFF`, `REDUCE_DOMAIN`) for branching

### Rewards
- `NNodes` -- change in number of explored nodes
- `SolvingTime` -- wall-clock time per step
- `LPIterations` -- change in LP iterations
- `DualIntegral` / `PrimalIntegral` / `PrimalDualIntegral` -- bound integrals
- `Done` -- binary completion status
- `ArithmeticMixin` -- compose rewards with Python operators: `-NNodes() * 0.5 + SolvingTime().cumsum()`

### Observations
- `NodeBipartite` / `NodeBipartiteEcole` -- pure-Python bipartite graph with configurable features
- `NodeBipartiteSCIP` -- thin wrapper around PySCIPOpt's built-in C implementation
- `Pseudocosts` -- per-variable pseudocost scores accumulated across branching history
- `StrongBranchingScores` -- full and partial strong-branching scores via LP probing

### Instance Generators
- `SetCoverGenerator` -- random set cover problems
- `IndependentSetGenerator` -- random independent set problems
- `CombinatorialAuctionGenerator` -- random combinatorial auction problems
- `CapacitatedFacilityLocationGenerator` -- random facility location problems
- `FileGenerator` -- load instances from files on disk

### Instance Modifiers
Modifiers wrap any generator and post-process the model before it reaches the environment:
- `EmbedObjective` -- embed the objective function as a variable + equality constraint
- `SetParameters` -- apply arbitrary SCIP parameters
- `SetNoCuts` / `SetNoHeuristics` / `SetNoDisplay` / `SetDFSNodeSelection` / `SetBFSNodeSelection` -- convenience presets

### Gymnasium Wrapper
`GymnasiumWrapper` exposes any `Environment` as a standard Gymnasium environment, compatible with Stable-Baselines3, CleanRL, and other RL frameworks.

## Tests

```bash
uv sync
uv run pytest
```

## Contributors

| Name | Affiliation | GitHub |
|------|-------------|--------|
| Olivier JUAN | EDF Lab | [@olivierjuan](https://github.com/olivierjuan) |
| Paul STRANG | EDF Lab · CNAM · ISAE | [@abfariah](https://github.com/abfariah) |

## License

[MIT](LICENSE)
