# Getting Started

## Prerequisites

Gyozas requires the [SCIP](https://www.scipopt.org/) solver (version 8+). But latest version should be installed when [PySCIPOpt](https://pyscipopt.readthedocs.io/en/latest/) is installed automatically with Gyozas dependencies

## Installation

Install gyozas from PyPI:

```bash
pip install "gyozas[viz]"
```

Or install from source for development:

```bash
git clone https://github.com/olivierjuan/gyozas.git
cd gyozas
pip install -e .[viz]
```

## Your First Episode

The code below creates a branching environment on random set cover instances and runs a single episode with a trivial "always pick the first candidate" policy.

```python
import gyozas

# 1. Create an instance generator
instances = gyozas.SetCoverGenerator(n_rows=50, n_cols=100, rng=42)

# 2. Build the environment
env = gyozas.Environment(
    instance_generator=instances,
    observation_function=gyozas.NodeBipartite(),
    reward_function=gyozas.NNodes(),
)

# 3. Reset starts a new episode (loads and solves a fresh instance)
obs, action_set, reward, done, info = env.reset()

total_reward = reward
steps = 0

# 4. Step until the solver finishes
while not done:
    action = action_set[0]  # naive policy
    obs, action_set, reward, done, info = env.step(action)
    total_reward += reward
    steps += 1

print(f"Episode finished in {steps} steps, total reward: {total_reward}")

# 5. Always close when done
env.close()
```

### Understanding the Output

Each call to `reset()` and `step()` returns five values:

| Value | Type | Description |
|-------|------|-------------|
| `obs` | array/tuple | Observation of the current solver state (e.g. bipartite graph features) |
| `action_set` | np.ndarray | Valid actions for the next step (variable indices for branching, node IDs for node selection) |
| `reward` | float | Reward signal for the transition |
| `done` | bool | Whether the episode has ended (solver finished) |
| `info` | dict/None | Additional information from the information function |

## Next Steps

- Read the [Concepts](concepts.md) page to understand how dynamics, rewards, and observations work together
- Try different [reward functions](api/rewards.md) like `SolvingTime` or `LPIterations`
- Use a different [instance generator](api/instances.md) like `IndependentSetGenerator`
