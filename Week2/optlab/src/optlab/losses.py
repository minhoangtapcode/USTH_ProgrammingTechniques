"""Pointwise losses: one class per likelihood.

Each is the negative log-likelihood of a distribution, up to constants:
squared error <-> Gaussian, logistic <-> Bernoulli, Poisson <-> Poisson counts.
Huber is not a likelihood but a robustification of the Gaussian one.
"""

import numpy as np

from .interfaces import IPointwiseLoss
from .types import Vec


class SquaredError(IPointwiseLoss):
    """[DAY 1] phi(z, y) = 1/2(z - y)^2. The Gaussian negative log-likelihood."""

    def value(self, z: Vec, y: Vec) -> Vec:
        return 0.5 * (z - y) ** 2
        

    def d1(self, z: Vec, y: Vec) -> Vec:
        return z - y

    def d2(self, z: Vec, y: Vec) -> Vec:
        return np.ones_like(z)


class LogisticNLL(IPointwiseLoss):
    """[DAY 1] phi(z, y) = log(1 + e^z) - y*z, for y in {0, 1}.

    Watch the overflow: `log(1 + exp(1000))` is `inf` computed naively and `1000`
    computed properly. Use the stable form of softplus. The trap is on the day-1 slide
    and in the tests.
    """
    def _softplus(self, z: Vec) -> Vec:
        # Stable computation of softplus: log(1 + exp(z))
        return np.maximum(0, z) + np.log1p(np.exp(-np.abs(z)))

    def _sigmoid(self, z: Vec) -> Vec:
        # Stable computation of sigmoid: 1 / (1 + exp(-z))
        exp_neg_z = np.exp(-np.abs(z))

    def value(self, z: Vec, y: Vec) -> Vec:
        return _softplus(z) - y * z

    def d1(self, z: Vec, y: Vec) -> Vec:
        return _sigmoid(z) - y
        

    def d2(self, z: Vec, y: Vec) -> Vec:
        s = _sigmoid(z)
        return s * (1.0 - s)
       


class Huber(IPointwiseLoss):
    """[DAY 6] Quadratic near zero, linear beyond `delta`.

    Bounded influence, so a gross outlier cannot dominate the fit - yet still smooth,
    unlike the L1 *penalty* of the same day. Note where each non-smoothness lives:
    Huber is smooth in the residual, L1 is non-smooth in the parameters.

    As `delta -> inf` it must reduce to `SquaredError`; a test checks exactly that.
    """

    def __init__(self, delta: float = 1.0) -> None:
        self.delta = delta

    def value(self, z: Vec, y: Vec) -> Vec:
        raise NotImplementedError("[DAY 6] lab 2")

    def d1(self, z: Vec, y: Vec) -> Vec:
        raise NotImplementedError("[DAY 6] lab 2")

    def d2(self, z: Vec, y: Vec) -> Vec:
        raise NotImplementedError("[DAY 6] lab 2")


class PoissonNLL(IPointwiseLoss):
    """[DAY 6] phi(z, y) = e^z - y*z. Counts with a log link.

    Writing this is the payoff of the open/closed principle: it is a new `IPointwiseLoss`,
    so gradient descent, Newton and ridge all work on it without a line changing anywhere
    else.
    """

    def value(self, z: Vec, y: Vec) -> Vec:
        raise NotImplementedError("[DAY 6] lab 2")

    def d1(self, z: Vec, y: Vec) -> Vec:
        raise NotImplementedError("[DAY 6] lab 2")

    def d2(self, z: Vec, y: Vec) -> Vec:
        raise NotImplementedError("[DAY 6] lab 2")
