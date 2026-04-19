from functools import partial
from logging import getLogger
from typing import Any

from pyscipopt import Model

from gyozas.instances import InstanceGenerator


class SetParameters(InstanceGenerator):
    """
    This class is a instance generator modifier. It sets SCIP solver parameters into the generated instances.
    """

    def __init__(self, instance_generator: InstanceGenerator, parameters: dict[str, Any]) -> None:
        """
        Initializes the SetParameters with an instance generator and the parameters dictionnary.

        :param instance_generator: The instance generator.
        :param parameters: The parameters to be set in the SCIP solver.
        """
        self.instance_generator = instance_generator
        self.parameters = parameters

    def seed(self, seed) -> None:
        self.instance_generator.seed(seed)

    def set_parameters(self, instance: Model) -> Model:
        logger = getLogger(__name__)
        for k, v in self.parameters.items():
            try:
                instance.setParam(k, v)
            except Exception as e:
                logger.warning(f"Parameter {k} could not be set due to {str(e)}")
        return instance

    def generate_instance(self, *args, **kwargs) -> Model:
        """
        Generates an instance and set parameters in the SCIP solver.

        :param args: Positional arguments for the instance generator.
        :param kwargs: Keyword arguments for the instance generator.
        :return: The generated instance.
        """
        return self.set_parameters(self.instance_generator.generate_instance(*args, **kwargs))

    def __next__(self) -> Model:
        model = self.instance_generator.__next__()
        return self.set_parameters(model)


NO_CUT_PARAMS = {
    "separating/poolfreq": -1,
    "separating/closecuts/freq": -1,
    "separating/flower/freq": -1,
    "separating/rlt/freq": -1,
    "separating/disjunctive/freq": -1,
    "separating/gauge/freq": -1,
    "separating/interminor/freq": -1,
    "separating/minor/freq": -1,
    "separating/convexproj/freq": -1,
    "separating/mixing/freq": -1,
    "separating/impliedbounds/freq": -1,
    "separating/intobj/freq": -1,
    "separating/cgmip/freq": -1,
    "separating/gomory/freq": -1,
    "separating/strongcg/freq": -1,
    "separating/gomorymi/freq": -1,
    "separating/flowcover/freq": -1,
    "separating/cmir/freq": -1,
    "separating/knapsackcover/freq": -1,
    "separating/aggregation/freq": -1,
    "separating/clique/freq": -1,
    "separating/clique/backtrackfreq": -1,
    "separating/zerohalf/freq": -1,
    "separating/lagromory/freq": -1,
    "separating/lagromory/cutgenfreq": -1,
    "separating/lagromory/cutaddfreq": -1,
    "separating/mcf/freq": -1,
    "separating/eccuts/freq": -1,
    "separating/oddcycle/freq": -1,
    "separating/rapidlearning/freq": -1,
}

NO_HEURISTIC_PARAMS = {
    "heuristics/dks/freq": -1,
    "heuristics/dps/freq": -1,
    "heuristics/padm/freq": -1,
    "heuristics/ofins/freq": -1,
    "heuristics/reoptsols/freq": -1,
    "heuristics/trivialnegation/freq": -1,
    "heuristics/trivial/freq": -1,
    "heuristics/clique/freq": -1,
    "heuristics/locks/freq": -1,
    "heuristics/vbounds/freq": -1,
    "heuristics/shiftandpropagate/freq": -1,
    "heuristics/zeroobj/freq": -1,
    "heuristics/completesol/freq": -1,
    "heuristics/dualval/freq": -1,
    "heuristics/repair/freq": -1,
    "heuristics/simplerounding/freq": -1,
    "heuristics/randrounding/freq": -1,
    "heuristics/zirounding/freq": -1,
    "heuristics/rounding/freq": -1,
    "heuristics/shifting/freq": -1,
    "heuristics/intshifting/freq": -1,
    "heuristics/oneopt/freq": -1,
    "heuristics/twoopt/freq": -1,
    "heuristics/indicator/freq": -1,
    "heuristics/scheduler/freq": -1,
    "heuristics/adaptivediving/freq": -1,
    "heuristics/indicatordiving/freq": -1,
    "heuristics/fixandinfer/freq": -1,
    "heuristics/farkasdiving/freq": -1,
    "heuristics/feaspump/freq": -1,
    "heuristics/conflictdiving/freq": -1,
    "heuristics/coefdiving/freq": -1,
    "heuristics/pscostdiving/freq": -1,
    "heuristics/fracdiving/freq": -1,
    "heuristics/nlpdiving/freq": -1,
    "heuristics/veclendiving/freq": -1,
    "heuristics/distributiondiving/freq": -1,
    "heuristics/intdiving/freq": -1,
    "heuristics/actconsdiving/freq": -1,
    "heuristics/objpscostdiving/freq": -1,
    "heuristics/rootsoldiving/freq": -1,
    "heuristics/linesearchdiving/freq": -1,
    "heuristics/guideddiving/freq": -1,
    "heuristics/octane/freq": -1,
    "heuristics/rens/freq": -1,
    "heuristics/alns/freq": -1,
    "heuristics/rins/freq": -1,
    "heuristics/localbranching/freq": -1,
    "heuristics/trustregion/freq": -1,
    "heuristics/gins/freq": -1,
    "heuristics/mutation/freq": -1,
    "heuristics/crossover/freq": -1,
    "heuristics/lpface/freq": -1,
    "heuristics/dins/freq": -1,
    "heuristics/bound/freq": -1,
    "heuristics/undercover/freq": -1,
    "heuristics/proximity/freq": -1,
    "heuristics/subnlp/freq": -1,
    "heuristics/mpec/freq": -1,
    "heuristics/multistart/freq": -1,
    "heuristics/trysol/freq": -1,
}

BRANCHING_RULE = {
    "branching/relpscost/priority": -536870912,
    "branching/pscost/priority": -536870912,
    "branching/inference/priority": -536870912,
    "branching/mostinf/priority": -536870912,
    "branching/leastinf/priority": -536870912,
    "branching/fullstrong/priority": -536870912,
    "branching/distribution/priority": -536870912,
    "branching/lookahead/priority": -536870912,
    "branching/multaggr/priority": -536870912,
    "branching/cloud/priority": -536870912,
    "branching/allfullstrong/priority": -536870912,
    "branching/gomory/priority": -536870912,
    "branching/vanillafullstrong/priority": -536870912,
    "branching/random/priority": -536870912,
    "branching/nodereopt/priority": -536870912,
}

NODE_SELECTION = {
    "nodeselection/estimate/stdpriority": -536870912,
    "nodeselection/estimate/memsavepriority": -536870912,
    "nodeselection/bfs/stdpriority": -536870912,
    "nodeselection/bfs/memsavepriority": -536870912,
    "nodeselection/hybridestim/stdpriority": -536870912,
    "nodeselection/hybridestim/memsavepriority": -536870912,
    "nodeselection/restartdfs/stdpriority": -536870912,
    "nodeselection/restartdfs/memsavepriority": -536870912,
    "nodeselection/uct/stdpriority": -536870912,
    "nodeselection/uct/memsavepriority": -536870912,
    "nodeselection/dfs/stdpriority": -536870912,
    "nodeselection/dfs/memsavepriority": -536870912,
    "nodeselection/breadthfirst/stdpriority": -536870912,
    "nodeselection/breadthfirst/memsavepriority": -536870912,
}


SetNoCuts = partial(SetParameters, parameters=NO_CUT_PARAMS)
SetNoHeuristics = partial(SetParameters, parameters=NO_HEURISTIC_PARAMS)
SetNoDisplay = partial(SetParameters, parameters={"display/verblevel": 0})
SetDFSNodeSelection = partial(
    SetParameters,
    parameters=NODE_SELECTION
    | {"nodeselection/dfs/stdpriority": 536870911, "nodeselection/dfs/memsavepriority": 536870911},
)
SetBFSNodeSelection = partial(
    SetParameters,
    parameters=NODE_SELECTION
    | {"nodeselection/bfs/stdpriority": 536870911, "nodeselection/bfs/memsavepriority": 536870911},
)
