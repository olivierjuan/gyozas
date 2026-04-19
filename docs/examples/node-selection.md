# Node Selection Agent Example

This example shows how to use `NodeSelectionDynamics` to control which node in the branch-and-bound tree to explore next.

## Basic Node Selection

```python
import gyozas

instances = gyozas.SetCoverGenerator(n_rows=50, n_cols=100, rng=0)

env = gyozas.Environment(
    instance_generator=instances,
    dynamics=gyozas.NodeSelectionDynamics(),
    reward_function=gyozas.NNodes(),
)

obs, action_set, reward, done, info = env.reset()
steps = 0

while not done:
    # action_set contains node IDs (integers)
    # A simple policy: always pick the first node
    action = action_set[0]
    obs, action_set, reward, done, info = env.step(action)
    steps += 1

print(f"Solved in {steps} node selections")
env.close()
```

## Depth-First vs Breadth-First

The node IDs in the action set correspond to SCIP's internal node numbering. You can implement different tree search strategies:

```python
import gyozas

instances = gyozas.SetCoverGenerator(n_rows=50, n_cols=100, rng=0)

env = gyozas.Environment(
    instance_generator=instances,
    dynamics=gyozas.NodeSelectionDynamics(),
    reward_function=gyozas.NNodes(),
)

obs, action_set, reward, done, info = env.reset()

while not done:
    # Higher node numbers tend to be deeper in the tree
    # Picking max approximates depth-first search
    action = max(action_set)
    obs, action_set, reward, done, info = env.step(action)

env.close()
```
