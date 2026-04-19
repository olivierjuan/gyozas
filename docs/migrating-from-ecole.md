# Migrating from Ecole

Gyozas is a pure-Python replacement for [Ecole](https://github.com/ds4dm/ecole). The APIs are intentionally similar, so migrating is straightforward.

## Why switch?

| | Ecole | Gyozas |
|---|---|---|
| **Installation** | Requires C++ compilation, CMake, custom SCIP build | `pip install gyozas` (only needs `libscip-dev`) |
| **Language** | C++ core with Python bindings | Pure Python |
| **Maintenance** | Unmaintained since 2023 | Actively maintained |
| **Python support** | Up to 3.10 | 3.10+ |
| **Dependencies** | Custom `ecole` C++ library | PySCIPOpt (pip-installable) |
| **Extensibility** | Subclass C++ base classes | Any object with `reset()`/`extract()` (duck typing via Protocol) |

## API Mapping

### Environment

```python
# Ecole
import ecole
env = ecole.environment.Branching(
    observation_function=ecole.observation.NodeBipartite(),
    reward_function=-ecole.reward.NNodes(),
    information_function=ecole.information.Nothing(),
    scip_params={"separating/maxrounds": 0},
)
instance = next(ecole.instance.SetCoverGenerator())
obs, action_set, reward, done, info = env.reset(instance)

# Gyozas
import gyozas
instances = gyozas.SetCoverGenerator()
env = gyozas.Environment(
    instance_generator=instances,  # pass the generator, not individual instances
    observation_function=gyozas.NodeBipartite(),
    reward_function=-gyozas.NNodes(),   # arithmetic works directly
    dynamics=gyozas.BranchingDynamics(),  # explicit dynamics parameter
    scip_params={"separating/maxrounds": 0},
)
obs, action_set, reward, done, info = env.reset()
```

**Key difference**: In Ecole, you pass a single instance to `reset()`. In gyozas, you pass the generator to the constructor and `reset()` automatically draws the next instance.

### Episode Loop

```python
# Ecole
obs, action_set, reward, done, info = env.reset(instance)
while not done:
    action = action_set[0]
    obs, action_set, reward, done, info = env.step(action)

# Gyozas -- identical!
obs, action_set, reward, done, info = env.reset()
while not done:
    action = action_set[0]
    obs, action_set, reward, done, info = env.step(action)
```

The 5-tuple return signature is the same: `(observation, action_set, reward, done, info)`.

!!! note "action_set type"
    In Ecole `action_set` is a list. In gyozas it is a `numpy.ndarray`.

### Dynamics

| Ecole | Gyozas | Notes |
|-------|--------|-------|
| `ecole.environment.Branching` | `gyozas.BranchingDynamics()` | Pass as `dynamics=` parameter |
| `ecole.environment.Configuring` | `gyozas.ConfiguringDynamics()` | Single-step parameter tuning |
| `ecole.environment.PrimalSearch` | `gyozas.PrimalSearchDynamics()` | LP-probing heuristic search |
| -- | `gyozas.NodeSelectionDynamics()` | New in gyozas |

### Observation Functions

| Ecole | Gyozas | Notes |
|-------|--------|-------|
| `ecole.observation.NodeBipartite()` | `gyozas.NodeBipartite()` | Pure-Python; returns `BipartiteGraph` dataclass |
| -- | `gyozas.NodeBipartiteSCIP()` | Uses PySCIPOpt's C implementation |
| `ecole.observation.StrongBranchingScores()` | `gyozas.StrongBranchingScores()` | Supports LP and pseudo candidates, configurable `itlim` |
| `ecole.observation.Pseudocosts()` | `gyozas.Pseudocosts()` | Tracks history across branching nodes |
| `ecole.observation.Khalil2016()` | -- | Not yet implemented |
| `ecole.observation.Nothing()` | Pass `observation_function=None` | |
| -- | `gyozas.MetaObservation([...])` | Combine multiple observations |

!!! note "BipartiteGraph return type"
    Ecole's `NodeBipartite` returns a named tuple. Gyozas returns a `BipartiteGraph` dataclass:
    ```python
    # Ecole
    col_feat, (edge_idx, edge_val), row_feat = obs

    # Gyozas
    col_feat  = obs.variable_features
    row_feat  = obs.row_features
    edge_idx  = obs.edge_features.indices
    edge_val  = obs.edge_features.values
    ```

### Reward Functions

| Ecole | Gyozas | Notes |
|-------|--------|-------|
| `ecole.reward.NNodes()` | `gyozas.NNodes()` | |
| `ecole.reward.LpIterations()` | `gyozas.LPIterations()` | |
| `ecole.reward.SolvingTime()` | `gyozas.SolvingTime()` | |
| `ecole.reward.IsDone()` | `gyozas.Done()` | |
| `ecole.reward.PrimalIntegral()` | `gyozas.PrimalIntegral()` | |
| `ecole.reward.DualIntegral()` | `gyozas.DualIntegral()` | |
| `ecole.reward.PrimalDualIntegral()` | `gyozas.PrimalDualIntegral()` | |
| `-1.5 * reward ** 2` (arithmetic) | `-1.5 * gyozas.NNodes() ** 2` | Fully supported via `ArithmeticMixin` |

### Information Functions

| Ecole | Gyozas | Notes |
|-------|--------|-------|
| `ecole.information.Nothing()` | `gyozas.Empty()` | Default |
| -- | `gyozas.informations.TimeSinceLastStep()` | Wall-clock delta per step |

### Instance Generators

| Ecole | Gyozas | Notes |
|-------|--------|-------|
| `ecole.instance.SetCoverGenerator()` | `gyozas.SetCoverGenerator()` | |
| `ecole.instance.IndependentSetGenerator()` | `gyozas.IndependentSetGenerator()` | |
| `ecole.instance.CombinatorialAuctionGenerator()` | `gyozas.CombinatorialAuctionGenerator()` | |
| `ecole.instance.CapacitatedFacilityLocationGenerator()` | `gyozas.CapacitatedFacilityLocationGenerator()` | |
| `ecole.instance.FileGenerator(...)` | `gyozas.FileGenerator(...)` | |

Seeding works the same way:

```python
# Ecole
generator = ecole.instance.SetCoverGenerator()
generator.seed(42)

# Gyozas -- same, or pass rng= to constructor
generator = gyozas.SetCoverGenerator(rng=42)
# or
generator.seed(42)
```

## Custom Components

In Ecole, custom functions require subclassing C++ base classes. In gyozas, any object with the right methods works (structural subtyping via `Protocol`):

```python
# Ecole -- must inherit from ecole.reward.RewardFunction (C++ class)
class MyReward(ecole.reward.RewardFunction):
    def before_reset(self, model):
        ...
    def extract(self, model, done):
        ...

# Gyozas -- just implement the methods, no inheritance needed
class MyReward:
    def reset(self, model):
        ...
    def extract(self, model, done):
        ...
```

!!! note "Method naming"
    Ecole uses `before_reset()` while gyozas uses `reset()`.

## Full Migration Example

```python
# --- Ecole ---
import ecole

generator = ecole.instance.SetCoverGenerator(n_rows=100, n_cols=200)
env = ecole.environment.Branching(
    observation_function=ecole.observation.NodeBipartite(),
    reward_function=ecole.reward.LpIterations(),
    scip_params={"separating/maxrounds": 0, "presolving/maxrounds": 0},
)

for _ in range(10):
    instance = next(generator)
    obs, action_set, reward, done, info = env.reset(instance)
    while not done:
        action = action_set[0]
        obs, action_set, reward, done, info = env.step(action)

# --- Gyozas ---
import gyozas

instances = gyozas.SetCoverGenerator(n_rows=100, n_cols=200)
env = gyozas.Environment(
    instance_generator=instances,
    observation_function=gyozas.NodeBipartite(),
    reward_function=gyozas.LPIterations(),
    scip_params={"separating/maxrounds": 0, "presolving/maxrounds": 0},
)

for _ in range(10):
    obs, action_set, reward, done, info = env.reset()
    while not done:
        action = action_set[0]
        obs, action_set, reward, done, info = env.step(action)

env.close()  # gyozas requires explicit cleanup
```

## What's Not Yet Ported

These Ecole features are not yet available in gyozas:

- `Khalil2016`, `Hutter2011`, `MilpBipartite` observation functions

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/olivierjuan/gyozas/blob/main/CONTRIBUTING.md).
