from pyscipopt import Model


class MetaObservation:
    """Combines multiple observation functions into a single composite observation.

    Parameters
    ----------
    observations
        A list, tuple, or dict of observation functions. If a dict is provided,
        ``extract()`` returns a dict keyed by the same names. Otherwise, it returns
        a container of the same type as the input.
    """

    def __init__(self, observations: list | dict | tuple) -> None:
        self.observations = observations

    def reset(self, model: Model) -> None:
        if isinstance(self.observations, dict):
            for _, obs in self.observations.items():
                obs.reset(model)
        else:
            for obs in self.observations:
                obs.reset(model)

    def extract(self, model: Model, done: bool) -> list | dict | tuple:
        if isinstance(self.observations, dict):
            return {name: obs.extract(model, done) for name, obs in self.observations.items()}
        else:
            type_ = type(self.observations)
            return type_(obs.extract(model, done) for obs in self.observations)
