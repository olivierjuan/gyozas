import numpy as np
from numpy.typing import NDArray
from pyscipopt import Model, quicksum

from gyozas.instances.instance_generator import InstanceGenerator, sanitize_rng


class CapacitatedFacilityLocationGenerator(InstanceGenerator):
    """Generator for random Capacitated Facility Location problem instances."""

    def __init__(
        self,
        n_customers=100,
        n_facilities=100,
        demand_interval=(5, 36),
        capacity_interval=(10, 161),
        fixed_cost_scale_interval=(100, 111),
        fixed_cost_cste_interval=(0, 91),
        ratio=5.0,
        continuous_assignment=True,
        rng=None,
    ) -> None:
        self.n_customers = n_customers
        self.n_facilities = n_facilities
        self.demand_interval = demand_interval
        self.capacity_interval = capacity_interval
        self.fixed_cost_scale_interval = fixed_cost_scale_interval
        self.fixed_cost_cste_interval = fixed_cost_cste_interval
        self.ratio = ratio
        self.continuous_assignment = continuous_assignment
        super().__init__(rng=rng)

    def __next__(self) -> Model:
        return self.generate_instance(
            n_customers=self.n_customers,
            n_facilities=self.n_facilities,
            demand_interval=self.demand_interval,
            capacity_interval=self.capacity_interval,
            fixed_cost_scale_interval=self.fixed_cost_scale_interval,
            fixed_cost_cste_interval=self.fixed_cost_cste_interval,
            ratio=self.ratio,
            continuous_assignment=self.continuous_assignment,
            rng=self.rng,
        )

    @staticmethod
    def _unit_transportation_costs(n_customers, n_facilities, rng) -> NDArray[np.float64]:
        scaling = 10.0
        customer_x = rng.random((n_customers, 1))
        customer_y = rng.random((n_customers, 1))
        facility_x = rng.random((1, n_facilities))
        facility_y = rng.random((1, n_facilities))
        costs = scaling * np.sqrt((customer_x - facility_x) ** 2 + (customer_y - facility_y) ** 2)
        assert costs.shape == (n_customers, n_facilities)
        return costs

    def generate_instance(
        self,
        n_customers=100,
        n_facilities=100,
        demand_interval=(5, 36),
        capacity_interval=(10, 161),
        fixed_cost_scale_interval=(100, 111),
        fixed_cost_cste_interval=(0, 91),
        ratio=5.0,
        continuous_assignment=True,
        rng=None,
    ) -> Model:
        rng = sanitize_rng(rng, default=self.rng)

        def randint(n, interval) -> NDArray[np.int64]:
            return rng.integers(interval[0], interval[1], size=n)

        # Generate data
        demands = randint(n_customers, demand_interval)
        capacities = randint(n_facilities, capacity_interval)
        fixed_costs = randint(n_facilities, fixed_cost_scale_interval) * np.sqrt(capacities) + randint(
            n_facilities, fixed_cost_cste_interval
        )
        transportation_costs = (
            CapacitatedFacilityLocationGenerator._unit_transportation_costs(n_customers, n_facilities, rng)
            * demands[:, np.newaxis]
        )

        # Scale capacities
        capacities = capacities * ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.rint(capacities)

        # Build SCIP model
        model = Model(f"CapacitatedFacilityLocation-{n_customers}-{n_facilities}")

        # Facility opening variables
        facility_vars = [model.addVar(vtype="BINARY", obj=fixed_costs[j], name=f"f_{j}") for j in range(n_facilities)]

        # Assignment variables
        serving_vars = np.empty((n_customers, n_facilities), dtype=object)
        for i in range(n_customers):
            for j in range(n_facilities):
                vtype = "CONTINUOUS" if continuous_assignment else "BINARY"
                serving_vars[i, j] = model.addVar(
                    vtype=vtype,
                    lb=0.0,
                    ub=1.0,
                    obj=transportation_costs[i, j],
                    name=f"s_{i}_{j}",
                )

        # Demand constraints
        for i in range(n_customers):
            model.addCons(
                quicksum(serving_vars[i, j] for j in range(n_facilities)) == 1.0,
                name=f"d_{i}",
            )

        # Capacity constraints
        for j in range(n_facilities):
            model.addCons(
                quicksum(serving_vars[i, j] * demands[i] for i in range(n_customers))
                <= capacities[j] * facility_vars[j],
                name=f"c_{j}",
            )

        # Tightening constraints
        total_demand = np.sum(demands)
        model.addCons(
            quicksum(facility_vars[j] * capacities[j] for j in range(n_facilities)) >= total_demand,
            name="t_total_demand",
        )
        for i in range(n_customers):
            for j in range(n_facilities):
                model.addCons(
                    serving_vars[i, j] <= facility_vars[j],
                    name=f"t_{i}_{j}",
                )

        model.setMinimize()
        return model
