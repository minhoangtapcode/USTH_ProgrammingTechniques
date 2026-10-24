"""[DAY 2] How far to move along a direction."""

from .errors import LineSearchFailed
from .interfaces import ILineSearch, IObjective
from .types import Vec


class FixedStep(ILineSearch):
    """A constant step length, ignoring the objective entirely.

    Safe when alpha <= 1/L for an L-smooth f; diverges above 2/L, which day 2 asks you
    to trigger on purpose and explain.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def step(self, objective: IObjective, x: Vec, g: Vec, d: Vec) -> float:
        return self.alpha

class Armijo(ILineSearch):
    """Backtracking until the sufficient-decrease condition holds:

        f(x + alpha*d) <= f(x) + c1*alpha*g^T d

    Start at `alpha0`, multiply by `rho` on rejection, give up after `max_backtracks`
    and raise `LineSearchFailed`. The optimizer catches that and reports
    `converged=False`; it must not crash.

    Injecting this in place of `FixedStep` must require no edit to `DescentOptimizer`.
    If it does, the loop is doing something that belongs here.
    """

    def __init__(
        self,
        alpha0: float = 1.0,
        c1: float = 1e-4,
        rho: float = 0.5,
        max_backtracks: int = 50,
    ) -> None:
        self.alpha0 = alpha0
        self.c1 = c1
        self.rho = rho
        self.max_backtracks = max_backtracks

    def step(self, objective: IObjective, x: Vec, g: Vec, d: Vec) -> float:
        slope = float(g @ d)
        if slope >= 0.0:
            raise ValueError(
                f"not a descent direction: g^T d = {slope:.6g} >= 0"
                "the direction rule, not the line search, is at fault"
            )
        f0 = objective.value(x)
        alpha = self.alpha0
        for _ in range(self.max_backtracks):
            if objective.value(x + alpha * d) <= f0 + self.c1 * alpha * slope:
                return alpha
            alpha *= self.rho

        raise LineSearchFailed(
            f"no step stastisfied the Armijo condition after {self.max_backtracks}"
              " backtracks (last alpha = {alpha:.3e}, g^t d = {slope:.3e})"
        )
        