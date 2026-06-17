# Probing Agent Example

`ProbingDynamics` turns strong-branching candidate selection into a *learnable* reliability
rule. At each node the agent receives the fractional LP branching candidates and returns a
**subset** to evaluate with strong branching (the expensive, exact probe). The unprobed
candidates are scored from pseudocosts, and the variable with the best combined score is
branched on. An empty subset falls back to pure pseudocost branching.

!!! warning "Pair it with a probing-cost reward"
    Strong branching is expensive, but that cost is only visible to the agent through the
    reward. Use `TotalLPIterations` (node LPs **and** strong-branch LPs) or
    `StrongBranchingLPIterations`. Plain `LPIterations` counts only node LPs and will *not*
    see the probing cost — with it, the optimal policy degenerates to "probe everything"
    (full strong branching).

!!! warning "Do not use `StrongBranchingScores` as the observation"
    `StrongBranchingScores` computes the full strong branching this dynamics exists to
    avoid, defeating the purpose. Use a per-variable observation such as
    `NodeBipartiteProbing`, whose features include how many times each variable has already
    been probed (`n_evaluations`) — exactly the reliability signal the agent needs.

## Raw loop (any Python agent)

Drive the environment directly. The action is a subset of `action_set` (the candidate LP
positions); here we probe every candidate, which recovers full strong branching:

```python
import gyozas

dyn = gyozas.ProbingDynamics(scoring="product")
env = gyozas.Environment(
    instance_generator=gyozas.SetCoverGenerator(n_rows=200, n_cols=100, rng=0),
    observation_function=gyozas.NodeBipartiteProbing(ledger=dyn.ledger),
    reward_function=-gyozas.NNodes() - 0.001 * gyozas.TotalLPIterations(),
    dynamics=dyn,
)

obs, action_set, reward, done, info = env.reset()
while not done:
    subset = action_set            # probe all candidates (replace with a learned policy)
    obs, action_set, reward, done, info = env.step(subset)
env.close()
```

Sharing the dynamics' `ledger` with `NodeBipartiteProbing` lets the observation expose each
variable's probing history as features.

## Training with Stable-Baselines3 / CleanRL

`ProbingDynamics` has a variable-length subset action, which RL libraries can't consume
directly. `ProbingGymnasiumWrapper` exposes it as a fixed `MultiBinary(K)` action over a
padded, masked candidate slate, with a `Dict` observation of per-candidate features plus a
validity `mask`:

```python
import gyozas
from gyozas.gymnasium_wrapper import ProbingGymnasiumWrapper

env = ProbingGymnasiumWrapper(
    instance_generator=gyozas.SetCoverGenerator(n_rows=200, n_cols=100, rng=0),
    reward_function=-gyozas.NNodes() - 0.001 * gyozas.TotalLPIterations(),
    max_candidates=1000,   # the fixed action-space size K
)

# from stable_baselines3 import PPO
# model = PPO("MultiInputPolicy", env, verbose=1)   # Dict obs -> MultiInputPolicy
# model.learn(total_timesteps=10_000)

obs, info = env.reset()
action = env.action_space.sample()   # MultiBinary(K): 1 = probe candidate i
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

`action[i] == 1` probes candidate `i` (padded/invalid slots are ignored via `obs["mask"]`).
By default the wrapper pairs `ProbingDynamics` with `NodeBipartiteProbing` automatically,
sharing their ledger. Nodes with more than `max_candidates` candidates are truncated to the
first `K` (set `on_overflow="error"` to forbid this).
