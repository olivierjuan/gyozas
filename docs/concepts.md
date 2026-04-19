# Concepts

Gyozas models the SCIP branch-and-bound solver as a **Partially Observable Markov Decision Process (POMDP)**. An RL agent interacts with the solver by making decisions (branching variable selection, node selection, algorithm configuration, or primal solution search) at each step, receiving observations and rewards.

## Architecture

The `Environment` class orchestrates four pluggable components:

```
Environment
  +-- Dynamics            (controls what decisions the agent makes)
  +-- ObservationFunction (extracts observations from solver state)
  +-- RewardFunction      (computes step rewards)
  +-- InformationFunction (returns auxiliary info)
  +-- InstanceGenerator   (yields SCIP Model instances)
```

## Episode Lifecycle

```
env.reset()
    |
    v
[Instance Generator] ---> new SCIP Model
    |
    v
[Dynamics.reset(model)] ---> starts solver in background thread
    |                         waits for first decision point
    v
observation, action_set, reward, done, info
    |
    v  (agent picks action)
env.step(action)
    |
    v
[Dynamics.step(action)] ---> sends action to solver thread
    |                         waits for next decision point
    v
observation, action_set, reward, done, info
    |
    v  (repeat until done=True)
env.close()
```

## Dynamics

Dynamics control **what type of decisions** the agent makes. Gyozas ships four:

### BranchingDynamics (default)

At each step, the agent selects which fractional variable to branch on. The `action_set` is a `numpy.ndarray` of variable indices (LP column positions) from SCIP's LP branching candidates. Optionally prepend extra sentinel actions (skip, cut off, reduce domain) via `with_extra_actions`.

```python
env = gyozas.Environment(
    instance_generator=instances,
    dynamics=gyozas.BranchingDynamics(),
)
```

### NodeSelectionDynamics

At each step, the agent selects which open node to explore next. The `action_set` is a `numpy.ndarray` of node IDs (leaves, children, siblings).

```python
env = gyozas.Environment(
    instance_generator=instances,
    dynamics=gyozas.NodeSelectionDynamics(),
)
```

### ConfiguringDynamics

Single-step dynamics for algorithm configuration. `reset()` returns immediately; the agent calls `step(param_dict)` once with a `dict` of SCIP parameter overrides, after which the solver runs to completion.

```python
env = gyozas.Environment(
    instance_generator=instances,
    dynamics=gyozas.ConfiguringDynamics(),
)
obs, _, reward, done, info = env.reset()
# agent chooses SCIP parameters
env.step({"separating/maxrounds": 0, "presolving/maxrounds": 5})
```

### PrimalSearchDynamics

At each heuristic call, the agent provides a partial variable assignment `(var_indices, vals)` over the current pseudo-branching candidates. The dynamics enters probing mode, fixes those variables, solves the LP, and tries the result as a feasible solution.

```python
env = gyozas.Environment(
    instance_generator=instances,
    dynamics=gyozas.PrimalSearchDynamics(trials_per_node=1),
)
obs, action_set, reward, done, info = env.reset()
import numpy as np
while not done:
    # pass empty assignment to skip this trial
    obs, action_set, reward, done, info = env.step(
        (np.array([], dtype=np.int64), np.array([], dtype=np.float64))
    )
```

### How Threaded Dynamics Work

`BranchingDynamics`, `NodeSelectionDynamics`, and `PrimalSearchDynamics` use **threading**: SCIP runs `model.optimize()` in a background daemon thread, and a custom PySCIPOpt plugin (Branchrule, Nodesel, or Heur) pauses execution at each decision point using `threading.Event` synchronisation. The main thread receives the action set, waits for the agent's action, then signals the solver to continue.

## Reward Functions

Reward functions compute a scalar signal at each step. They follow a simple protocol:

```python
class RewardFunction(Protocol):
    def reset(self, model: Model) -> None: ...
    def extract(self, model: Model, done: bool) -> float: ...
```

Built-in rewards:

| Class | Signal |
|-------|--------|
| `NNodes` | Change in number of explored nodes |
| `SolvingTime` | Wall-clock time since last step |
| `LPIterations` | Change in LP iteration count |
| `DualIntegral` | Dual bound integral over time |
| `PrimalIntegral` | Primal bound integral over time |
| `PrimalDualIntegral` | Sum of primal and dual integrals |
| `Done` | 1.0 if optimal solution found, 0.0 otherwise |

### Reward Arithmetic

All built-in reward classes inherit from `ArithmeticMixin`, which overloads the standard arithmetic operators so reward functions can be freely composed:

```python
# Negate node count and add a time penalty
reward = -gyozas.NNodes() + gyozas.SolvingTime() * 0.1

# Accumulate LP iterations over the episode
reward = gyozas.LPIterations().cumsum()

# Square root of primal-dual integral
reward = gyozas.PrimalDualIntegral().sqrt()
```

Every operator returns a new `ArithmeticMixin` that satisfies the `RewardFunction` protocol, so composed rewards work everywhere plain rewards do. The `cumsum()` method wraps a reward in a running sum that resets at episode boundaries.

## Observation Functions

Observation functions extract features from the solver state. They follow the same protocol pattern:

```python
class ObservationFunction(Protocol):
    def reset(self, model: Model) -> None: ...
    def extract(self, model: Model, done: bool) -> Any: ...
```

Built-in observations:

| Class | Output |
|-------|--------|
| `NodeBipartite` / `NodeBipartiteEcole` | Pure-Python bipartite graph with configurable features |
| `NodeBipartiteSCIP` | Bipartite graph via PySCIPOpt's C implementation |
| `Pseudocosts` | Per-variable pseudocost scores (1-D array, NaN for non-candidates) |
| `StrongBranchingScores` | Full or partial strong-branching scores via LP probing |
| `MetaObservation` | Combines multiple observation functions into a tuple |

### BipartiteGraph

All bipartite graph observations return a `BipartiteGraph` dataclass:

```python
@dataclass
class EdgeFeatures:
    indices: np.ndarray   # shape (2, n_edges) — [row_idx, col_idx]
    values: np.ndarray    # shape (n_edges,)   — edge coefficients

@dataclass
class BipartiteGraph:
    variable_features: np.ndarray  # shape (n_vars, n_col_features)
    row_features: np.ndarray       # shape (n_rows, n_row_features)
    edge_features: EdgeFeatures
```

Access fields by name:

```python
obs, action_set, reward, done, info = env.reset()
col_features = obs.variable_features   # (n_vars, n_col_features)
row_features = obs.row_features        # (n_rows, n_row_features)
edge_idx     = obs.edge_features.indices
edge_vals    = obs.edge_features.values
```

The bipartite graph representation (from [Gasse et al., NeurIPS 2019](https://arxiv.org/abs/1906.01629)) models the LP relaxation as a bipartite graph between constraint rows and variable columns.

## Instance Generators

Instance generators are iterators that yield PySCIPOpt `Model` objects:

```python
instances = gyozas.SetCoverGenerator(n_rows=100, n_cols=200, rng=42)
model = next(instances)  # returns a pyscipopt.Model
```

Built-in generators:

| Class | Problem |
|-------|---------|
| `SetCoverGenerator` | Weighted set cover |
| `IndependentSetGenerator` | Maximum independent set |
| `CombinatorialAuctionGenerator` | Combinatorial auction winner determination |
| `CapacitatedFacilityLocationGenerator` | Capacitated facility location |
| `FileGenerator` | Load instances from `.mps`/`.lp` files on disk |

All generators support seeding via `rng` parameter or `.seed()` method for reproducibility.

## Custom Components

You can create custom rewards, observations, or information functions by implementing the protocol:

```python
class MyReward:
    def reset(self, model):
        self.prev_gap = 1.0

    def extract(self, model, done):
        gap = model.getGap()
        delta = self.prev_gap - gap
        self.prev_gap = gap
        return delta

env = gyozas.Environment(
    instance_generator=instances,
    reward_function=MyReward(),
)
```

No inheritance required — any object with `reset()` and `extract()` methods works (structural subtyping via `Protocol`).
