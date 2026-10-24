"""[DAY 2] When the loop ends."""

from .interfaces import IStoppingCriterion
from .results import StepEvent


class GradientNormBelow(IStoppingCriterion):
    """Stop once ||grad f|| <= tol. The only criterion here that means genuine convergence."""
    means_convergence = True
    def __init__(self, tol: float = 1e-6) -> None:
        self.tol = tol

    def should_stop(self, event: StepEvent) -> bool:
        return event.grad_norm <= self.tol

class MaxIterations(IStoppingCriterion):
    """Stop after a budget. Firing this means `converged=False`, not success."""

    def __init__(self, max_iter: int = 1000) -> None:
        self.max_iter = max_iter

    def should_stop(self, event: StepEvent) -> bool:
        return event.iteration >= self.max_iter


class AnyOf(IStoppingCriterion):
    """Stop as soon as any of the wrapped criteria fires.

    A composite, so the loop still sees exactly one `IStoppingCriterion` no matter how
    many conditions you combine.
    """
    means_convergence = False
    def __init__(self, *criteria: IStoppingCriterion) -> None:
        self.criteria = criteria
        self.means_convergence = False

    def should_stop(self, event: StepEvent) -> bool:
        fired = [c for c in self.criteria if c.should_stop(event)]
        self.means_convergence = any(
            getattr(c, "means_convergence", False) for c in fired
        )
        return bool(fired)