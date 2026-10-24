"""[DAY 2] The one deterministic descent loop, written once and never again.

Gradient descent, momentum, Newton and ridge are all this loop with different objects
plugged in. If you find yourself editing it on day 4 to make Newton work, something
belongs in an `IDirectionRule` or an `ILineSearch` instead.
"""

import numpy as np

from ..errors import LineSearchFailed
from ..interfaces import (
    IDirectionRule,
    ILineSearch,
    IObjective,
    IObserver,
    IOptimizer,
    IStoppingCriterion,
)
from ..linalg import CholeskySolver
from ..linesearch import Armijo, FixedStep
from ..results import OptimizeResult, StepEvent
from ..stopping import AnyOf, GradientNormBelow, MaxIterations
from ..types import Vec
from .directions import HeavyBall, NewtonDirection, SteepestDescent


def _fired_on_convergence(stop: IStoppingCriterion) -> bool:
    """Return whether the stopping criterion fired for genuine convergence."""
    return bool(getattr(stop, "means_convergence", False))


class DescentOptimizer(IOptimizer):
    """x <- x + alpha*d, where `direction` chooses d and `line_search` chooses alpha.

    Every collaborator arrives through the constructor. The loop itself only:
    evaluates, asks for a direction, asks for a step, moves, builds a `StepEvent`,
    notifies the observers, and asks the criterion whether to stop.

    Catch `LineSearchFailed` and return `converged=False` with the reason in `message`.
    """
    def _fired_on_convergence(stop: IStoppingCriterion) -> bool:
        """Return True if the stopping criterion that fired means convergence.

        The loop sees only one `IStoppingCriterion`, but it may be a composite of many.
        """
        return getattr(stop, "means_convergence", False)
    def __init__(
        self,
        direction: IDirectionRule,
        line_search: ILineSearch,
        stop: IStoppingCriterion,
        observers: list[IObserver] | None = None,
    ) -> None:
        self.direction = direction
        self.line_search = line_search
        self.stop = stop
        self.observers = observers or []

    def minimize(self, objective: object, x0: Vec) -> OptimizeResult:
        """Requires an `IObjective`."""
        if not isinstance(objective, IObjective):
            raise TypeError(
                f"DescentOptimizer requires an IObjective, got {type(objective).__name__}"
            )

        x = np.array(x0, dtype=np.float64).copy()
        value = objective.value(x)
        g = objective.gradient(x)
        grad_norm = float(np.linalg.norm(objective.gradient(x)))
        iteration = 0

        while True:
            g = objective.gradient(x)
            d = self.direction.direction(objective, x, g)
            try:
                alpha = self.line_search.step(objective, x, g, d)
            except LineSearchFailed as err:
                return OptimizeResult(
                    x=x,
                    value=value,
                    grad_norm=grad_norm,
                    iterations=iteration,
                    converged=False,
                    message=str(err),
                )

            x = x + alpha * d
            iteration += 1
            value = objective.value(x)
            grad_norm = float(np.linalg.norm(objective.gradient(x)))
            event = StepEvent(
                iteration=iteration,
                x=x,
                value=value,
                grad_norm=grad_norm,
                step_size=alpha,
            )

            for observer in self.observers:
                observer.on_step(event)

            if self.stop.should_stop(event):
                converged = _fired_on_convergence(self.stop)
                return OptimizeResult(
                    x=x,
                    value=value,
                    grad_norm=grad_norm,
                    iterations=iteration,
                    converged=converged,
                    message="" if converged else "stopping criterion fired",
                    )
                
def gradient_descent(tol: float = 1e-6, max_iter: int = 1000) -> DescentOptimizer:
    """[DAY 2] Steepest descent with an Armijo line search."""
    return DescentOptimizer(
        direction=SteepestDescent(),
        line_search=Armijo(),
        stop=AnyOf(GradientNormBelow(tol), MaxIterations(max_iter)),
    )


def momentum(beta: float = 0.9, alpha: float = 0.01, max_iter: int = 1000) -> DescentOptimizer:
    """[DAY 2] Heavy ball with a fixed step."""
    return DescentOptimizer(
        direction=HeavyBall(beta),
        line_search=FixedStep(alpha),
        stop=AnyOf(GradientNormBelow(), MaxIterations(max_iter)),
    )


def newton(tol: float = 1e-8, max_iter: int = 100) -> DescentOptimizer:
    """[DAY 4] Newton's direction, Cholesky solver, Armijo starting at alpha = 1.

    Note what this function is: four existing objects composed. No new loop, no new
    optimizer class. If it reads as anything more than wiring, the wiring is wrong.
    """
    raise NotImplementedError("[DAY 4] lab 2")
