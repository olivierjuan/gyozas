"""Gyozas - Reinforcement Learning for Combinatorial Optimization."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gyozas")
except PackageNotFoundError:
    __version__ = "unknown"

from gyozas.collection import CollectionMetadata
from gyozas.dynamics import Dynamics
from gyozas.dynamics.branching import (
    BranchingDecision,
    BranchingDirection,
    BranchingDynamics,
    ExtraBranchingActions,
    PairedCounters,
)
from gyozas.dynamics.node_selection import NodeSelectionDynamics
from gyozas.environment import Environment
from gyozas.experts import BranchingExpert, PseudocostExpert, RelpscostExpert, StrongBranchingExpert
from gyozas.gymnasium_wrapper import GymnasiumWrapper
from gyozas.informations.empty import Empty
from gyozas.informations.solve_status import SolveStatus
from gyozas.instances import InstanceGenerator
from gyozas.instances.capacitated_facility_location import CapacitatedFacilityLocationGenerator
from gyozas.instances.combinatorial_auction import CombinatorialAuctionGenerator
from gyozas.instances.files import FileGenerator
from gyozas.instances.independent_set import IndependentSetGenerator
from gyozas.instances.multiple_knapsack import MultipleKnapsackGenerator
from gyozas.instances.set_cover import SetCoverGenerator
from gyozas.metrics import (
    catastrophic_losses,
    friedman_nemenyi,
    paired_per_instance_ratios,
    quantiles,
    save_table,
    sgm_of_ratios,
    shifted_geometric_mean,
    solved_vs_truncated,
    summary_table,
    wilcoxon_holm,
    wilcoxon_pair,
    win_rate,
)
from gyozas.observations import NodeBipartite, NodeBipartiteEcole, NodeBipartiteSCIP, SolverStats
from gyozas.observations.meta_observation import MetaObservation
from gyozas.rewards.done import Done
from gyozas.rewards.integral_bound import DualIntegral, PrimalDualIntegral, PrimalIntegral
from gyozas.rewards.lp_iterations import LPIterations
from gyozas.rewards.nnodes import NNodes
from gyozas.rewards.solving_time import SolvingTime
from gyozas.seeds import SeedTriplet
from gyozas.solver_config import SolverConfig, available_profiles, get_profile, register_profile
from gyozas.trajectories import (
    Trajectory,
    TrajectoryRecorder,
    load_trajectories,
    retrospective_trajectories,
    save_trajectories,
)

__all__ = [
    "BranchingDecision",
    "BranchingExpert",
    "PseudocostExpert",
    "RelpscostExpert",
    "StrongBranchingExpert",
    "Trajectory",
    "TrajectoryRecorder",
    "load_trajectories",
    "retrospective_trajectories",
    "save_trajectories",
    "BranchingDirection",
    "BranchingDynamics",
    "PairedCounters",
    "CapacitatedFacilityLocationGenerator",
    "CollectionMetadata",
    "CombinatorialAuctionGenerator",
    "Done",
    "DualIntegral",
    "Dynamics",
    "Empty",
    "Environment",
    "ExtraBranchingActions",
    "FileGenerator",
    "GymnasiumWrapper",
    "IndependentSetGenerator",
    "InstanceGenerator",
    "LPIterations",
    "MetaObservation",
    "MultipleKnapsackGenerator",
    "NNodes",
    "NodeBipartite",
    "NodeBipartiteSCIP",
    "NodeBipartiteEcole",
    "NodeSelectionDynamics",
    "PrimalDualIntegral",
    "PrimalIntegral",
    "SetCoverGenerator",
    "SeedTriplet",
    "SolvingTime",
    "SolverConfig",
    "SolverStats",
    "SolveStatus",
    "available_profiles",
    "catastrophic_losses",
    "friedman_nemenyi",
    "get_profile",
    "paired_per_instance_ratios",
    "quantiles",
    "register_profile",
    "save_table",
    "shifted_geometric_mean",
    "solved_vs_truncated",
    "sgm_of_ratios",
    "summary_table",
    "wilcoxon_holm",
    "wilcoxon_pair",
    "win_rate",
]
