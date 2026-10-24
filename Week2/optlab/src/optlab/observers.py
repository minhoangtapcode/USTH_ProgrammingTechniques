"""[DAY 2] Watching the loop without changing it."""

from .interfaces import IObserver
from .results import StepEvent


class History(IObserver):
    """Records every `StepEvent`, for the convergence plots and the rate checks.

    Recording is separate from iterating: the optimizer does not know whether anyone is
    watching, and adding a second observer changes nothing in the loop.
    """

    def __init__(self) -> None:
        self.events: list[StepEvent] = []

    def on_step(self, event: StepEvent) -> None:
        self.events.append(event)

    @property
    def values(self) -> list[float]:
        """The objective value at each iteration."""
        return [event.value for event in self.events]

    @property
    def grad_norms(self) -> list[float]:
        """||grad f|| at each iteration - plot this on a log scale to read the rate."""
        return [event.grad_norm for event in self.events]
