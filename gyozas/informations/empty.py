from pyscipopt import Model


class Empty:
    """Information function that returns no additional information."""

    def reset(self, model: Model) -> None:
        pass

    def extract(self, model: Model, done: bool) -> None:
        pass
