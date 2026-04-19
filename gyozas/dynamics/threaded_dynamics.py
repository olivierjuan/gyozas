from threading import Event, Thread

from pyscipopt import Model

from gyozas.dynamics.dynamics import Dynamics


class ThreadedDynamics(Dynamics):
    """Mixin that manages the solve thread shared by all threaded dynamics.

    Provides:
    - ``obs_event``, ``action_event``, ``die_event`` — synchronisation primitives
    - ``_start_solve_thread(model)`` — starts ``model.optimize()`` in a daemon
      thread; guarantees ``obs_event`` is set when the thread exits (even on error)
    - ``close()`` — signals ``die_event`` and joins with a timeout
    - ``__del__`` — calls ``close()`` safely
    """

    def __init__(self) -> None:
        super().__init__()
        self.obs_event: Event = Event()
        self.action_event: Event = Event()
        self.die_event: Event = Event()
        self.done: bool = False
        self._thread: Thread | None = None

    def _start_solve_thread(self, model: Model) -> None:
        """Run ``model.optimize()`` in a background daemon thread."""

        def go() -> None:
            try:
                model.optimize()
            finally:
                self.done = True
                self.obs_event.set()

        self._thread = Thread(target=go, daemon=True)
        self._thread.start()

    def _stop_thread(self) -> None:
        """Signal the current solve thread to exit and wait for it to finish."""
        if hasattr(self, "die_event"):
            self.die_event.set()
        if hasattr(self, "action_event"):
            # Unblock the oracle if it is waiting for an action.
            self.action_event.set()
        if hasattr(self, "_thread") and self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        self._thread = None

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        self._stop_thread()
