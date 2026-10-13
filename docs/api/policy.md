# Policy Interface

The single interface every branching policy exposes: given the current
observation (whatever views the policy declared — bipartite graph, solver
stats, RL observation, or a composite) and the admissible action set at the
node, return one action. The harness drives the solver, never the inverse;
a policy can be architecturally anything.

A new policy is one class (or one function) plus the observation it declares —
no harness change.

::: gyozas.policy.Policy

::: gyozas.policy.CallablePolicy
