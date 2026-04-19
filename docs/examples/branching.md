# Branching Agent Example

This example shows how to build a simple branching agent that selects the variable with the strongest pseudocost.

## Random Branching

The simplest possible agent picks a random action at each step:

```python
import random
import gyozas

instances = gyozas.SetCoverGenerator(n_rows=100, n_cols=200, rng=0)

env = gyozas.Environment(
    instance_generator=instances,
    observation_function=gyozas.NodeBipartite(),
    reward_function=gyozas.NNodes(),
)

obs, action_set, reward, done, info = env.reset()
total_reward = reward

while not done:
    action = random.choice(action_set)
    obs, action_set, reward, done, info = env.step(action)
    total_reward += reward

print(f"Total nodes explored: {total_reward}")
env.close()
```

## Using Observations

`NodeBipartite` returns a `BipartiteGraph` dataclass. Access features via named attributes:

```python
import numpy as np
import gyozas

instances = gyozas.SetCoverGenerator(n_rows=100, n_cols=200, rng=0)

env = gyozas.Environment(
    instance_generator=instances,
    observation_function=gyozas.NodeBipartite(),
    reward_function=gyozas.NNodes(),
)

obs, action_set, reward, done, info = env.reset()

while not done:
    col_features = obs.variable_features  # shape (n_vars, n_col_features)
    # Column feature index 4 is solution fractionality in NodeBipartiteEcole
    fracs = col_features[action_set, 4]
    action = action_set[int(np.argmax(fracs))]
    obs, action_set, reward, done, info = env.step(action)

env.close()
```

## Using Pseudocosts

`Pseudocosts` accumulates branching history across nodes and returns a 1-D score per LP column (NaN for non-candidates):

```python
import numpy as np
import gyozas
from gyozas.observations import Pseudocosts

instances = gyozas.SetCoverGenerator(n_rows=100, n_cols=200, rng=0)

env = gyozas.Environment(
    instance_generator=instances,
    observation_function=Pseudocosts(),
    reward_function=gyozas.NNodes(),
)

obs, action_set, reward, done, info = env.reset()

while not done:
    # obs is a 1-D array; pick the candidate with the highest pseudocost score
    scores = obs[action_set]
    action = action_set[int(np.argmax(np.nan_to_num(scores, nan=-np.inf)))]
    obs, action_set, reward, done, info = env.step(action)

env.close()
```

## Composed Rewards

Reward functions support full arithmetic composition via `ArithmeticMixin`:

```python
import gyozas

instances = gyozas.SetCoverGenerator(n_rows=50, n_cols=100, rng=42)

env = gyozas.Environment(
    instance_generator=instances,
    # Penalise nodes and LP iterations jointly
    reward_function=-(gyozas.NNodes() + gyozas.LPIterations() * 0.01),
)

obs, action_set, reward, done, info = env.reset()
while not done:
    action = action_set[0]
    obs, action_set, reward, done, info = env.step(action)

env.close()
```

## Multiple Episodes

Run multiple episodes to collect training data:

```python
import gyozas

instances = gyozas.SetCoverGenerator(n_rows=50, n_cols=100, rng=42)

env = gyozas.Environment(
    instance_generator=instances,
    reward_function=gyozas.LPIterations(),
)

for episode in range(5):
    obs, action_set, reward, done, info = env.reset()
    episode_reward = reward
    steps = 0

    while not done:
        action = action_set[0]
        obs, action_set, reward, done, info = env.step(action)
        episode_reward += reward
        steps += 1

    print(f"Episode {episode}: {steps} steps, reward={episode_reward:.1f}")

env.close()
```
