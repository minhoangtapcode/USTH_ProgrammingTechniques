"""[DAY 3] Training on a sample of the gradient - how modern machine learning actually fits.

These are their own `IOptimizer`s rather than `IDirectionRule`s: the loop is different
(epochs, shuffling, no line search), so forcing them into `DescentOptimizer` would mean
editing it. A new implementation of an existing interface is the open/closed answer.

Both depend only on `IBatchObjective`. Neither asks for a Hessian, a line search, or a
full gradient.
"""

import numpy as np

from ..interfaces import IBatchObjective, IObserver, IOptimizer, IStoppingCriterion
from ..results import OptimizeResult, StepEvent
from ..types import Vec

#def _batches(rng: np.random.Generator, n: int, batch_sizes: int) -> list[Index]:

 #   if batch_sizes < 1:
 #       raise

class SGD(IOptimizer):
    """Mini-batch stochastic gradient descent.

    The random generator is injected, never created inside: that is what makes a run
    reproducible from a seed, and it is dependency inversion applied to randomness.

    With `batch_size = n` this must reduce exactly to gradient descent with a fixed
    step - there is a test for it.

    A constant step converges only to a noise floor proportional to alpha; a decaying
    schedule satisfying the Robbins-Monro conditions converges properly. Day 3 asks you
    to see both.
    """

    def __init__(
        self,
        batch_size: int = 32,
        lr: float = 0.01,
        n_epochs: int = 50,
        momentum: float = 0.0,
        lr_decay: float = 0.0,
        rng: np.random.Generator | None = None,
        observers: list[IObserver] | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.momentum = momentum
        self.lr_decay = lr_decay
        self.rng = rng if rng is not None else np.random.default_rng()
        self.observers = observers or []

    def minimize(self, objective: object, x0: Vec) -> OptimizeResult:
        """Requires an `IBatchObjective`."""
        if not isinstance(objective, IBatchObjective):
            raise TypeError(
                f"SGD requires an IBatchObjective, got {type(objective).__name__}"
            )

        x = np.array(x0, dtype=np.float64).copy()
        velocity = np.zeros_like(x)
        n_samples = objective.n_samples

        for epoch in range(self.n_epochs):
            order = self.rng.permutation(n_samples)
            step_size = self.lr / (1.0 + self.lr_decay * epoch)

            for start in range(0, n_samples, self.batch_size):
                idx = order[start:start + self.batch_size]
                gradient = objective.batch_gradient(x, idx)
                velocity = self.momentum * velocity - step_size * gradient
                x = x + velocity

            value = float(objective.value(x))
            grad_norm = float(np.linalg.norm(objective.gradient(x)))
            event = StepEvent(
                iteration=epoch + 1,
                x=x.copy(),
                value=value,
                grad_norm=grad_norm,
                step_size=step_size,
            )
            for observer in self.observers:
                observer.on_step(event)

        return OptimizeResult(
            x=x,
            value=float(objective.value(x)),
            grad_norm=float(np.linalg.norm(objective.gradient(x))),
            iterations=self.n_epochs,
            converged=False,
            message="epoch budget exhausted",
        )


class Adam(IOptimizer):
    """Adaptive moment estimation: a per-coordinate step size from the running moments.

    Keeps m (first moment) and v (second), both bias-corrected because they start at
    zero and would otherwise be biased toward it for the first several steps.

    What it really does is auto-standardize the per-coordinate scale - the optimizer-side
    answer to the conditioning problem that day 1 diagnosed in the data. On the ill-scaled
    (1 vs 1000) problem it needs far fewer epochs than SGD; that contrast is the lab.

    Its very first step is approximately lr*sign(g), which is a good test.
    """

    def __init__(
        self,
        batch_size: int = 32,
        lr: float = 0.001,
        n_epochs: int = 50,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        rng: np.random.Generator | None = None,
        observers: list[IObserver] | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.rng = rng if rng is not None else np.random.default_rng()
        self.observers = observers or []

    def minimize(self, objective: object, x0: Vec) -> OptimizeResult:
        """Requires an `IBatchObjective`."""
        if not isinstance(objective, IBatchObjective):
            raise TypeError(
                f"Adam requires an IBatchObjective, got {type(objective).__name__}"
            )

        x = np.array(x0, dtype=np.float64).copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        n_samples = objective.n_samples
        step_count = 0

        for epoch in range(self.n_epochs):
            order = self.rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                idx = order[start:start + self.batch_size]
                gradient = objective.batch_gradient(x, idx)
                step_count += 1
                m = self.beta1 * m + (1.0 - self.beta1) * gradient
                v = self.beta2 * v + (1.0 - self.beta2) * gradient**2
                m_hat = m / (1.0 - self.beta1**step_count)
                v_hat = v / (1.0 - self.beta2**step_count)
                x = x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            value = float(objective.value(x))
            grad_norm = float(np.linalg.norm(objective.gradient(x)))
            event = StepEvent(
                iteration=epoch + 1,
                x=x.copy(),
                value=value,
                grad_norm=grad_norm,
                step_size=self.lr,
            )
            for observer in self.observers:
                observer.on_step(event)

        return OptimizeResult(
            x=x,
            value=float(objective.value(x)),
            grad_norm=float(np.linalg.norm(objective.gradient(x))),
            iterations=self.n_epochs,
            converged=False,
            message="epoch budget exhausted",
        )
