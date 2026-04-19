from pyscipopt import Model, quicksum

from gyozas.instances import InstanceGenerator


class EmbedObjective(InstanceGenerator):
    """
    This class is a instance generator modifier. It embeds an objective function into the generated instances.
    It modifies the instance generator to include a variable representing the objective function,
    and adds a constraint that ensures the new variable is equal to the old objective function.
    It then modifies the objective to be the new variable, effectively embedding the objective into the instance.
    """

    def __init__(self, instance_generator: InstanceGenerator, replace_fobj=False) -> None:
        """
        Initializes the EmbedObjective with an instance generator.

        :param instance_generator: The instance generator to be embedded in the system.
        """
        self.instance_generator = instance_generator
        self.replace_fobj = replace_fobj

    def seed(self, seed) -> None:
        self.instance_generator.seed(seed)

    def generate_instance(self, *args, **kwargs) -> Model:
        """
        Generates an instance using the embedded instance generator.

        :param args: Positional arguments for the instance generator.
        :param kwargs: Keyword arguments for the instance generator.
        :return: The generated instance.
        """
        return self.instance_generator.generate_instance(*args, **kwargs)

    def __next__(self) -> Model:
        model = self.instance_generator.__next__()
        # Capture original objective sense and coefficients before adding _fobj_
        sense = "minimize" if model.getObjectiveSense() == "minimize" else "maximize"
        variables = model.getVars()
        obj_terms = [(var, var.getObj()) for var in variables if var.getObj() != 0]
        _fobj_ = model.addVar(vtype="C", lb=None, ub=None, name="_fobj_")
        # Constrain _fobj_ to equal the original objective
        model.addCons(
            quicksum(coef * var for var, coef in obj_terms) == _fobj_,
            name="objective_function_constraint",
        )
        if self.replace_fobj:
            model.setObjective(_fobj_, sense=sense)
        return model
