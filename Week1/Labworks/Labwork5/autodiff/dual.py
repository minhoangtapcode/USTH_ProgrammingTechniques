"""Forward mode: dual numbers a + b·ε with ε² = 0.

Part 1 of the project. Replace every `raise NotImplementedError` by an implementation.
`__add__` is given as a worked example.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Dual:
    """A dual number `val + der·ε`: a value and its derivative travelling together."""

    val: float
    der: float = 0.0

    @staticmethod
    def lift(x: Dual | float) -> Dual:
        """Turn a plain number into a constant dual number (its derivative is 0)."""
        if isinstance(x, Dual):
            return x
        return Dual(float(x), 0.0)

    # ---- arithmetic ------------------------------------------------------------
    def __add__(self, other: Dual | float) -> Dual:
        """Worked example: (a + bε) + (c + dε) = (a + c) + (b + d)ε."""
        if not isinstance(other, (Dual, int, float)):
            return NotImplemented          # lets Python raise the proper TypeError
        o = Dual.lift(other)
        return Dual(self.val + o.val, self.der + o.der)

    def __radd__(self, other: float) -> Dual:
        """Called for `2 + x` when `x` is a Dual (addition is commutative)."""
        return self + other

    def __neg__(self) -> Dual:
        return Dual(-self.val, -self.der)

    def __sub__(self, other: Dual | float) -> Dual:
        if not isinstance(other, (Dual, int, float)):
            return NotImplemented 
        return self + (-Dual.lift(other))        

    def __rsub__(self, other: float) -> Dual:
        """Called for `2 - x`: careful, subtraction is NOT commutative."""
        return Dual.lift(other) - self

    def __mul__(self, other: Dual | float) -> Dual:
        """(a + bε)(c + dε) = ac + (ad + bc)ε  - the product rule!"""
        if not isinstance(other, (Dual, int, float)):
            return NotImplemented
        
        o = Dual.lift(other)
        value = self.val * o.val
        derivative = self.der * o.val + self.val * o.der
        return Dual(value, derivative)

    def __rmul__(self, other: float) -> Dual:
        return self * other

    def __truediv__(self, other: Dual | float) -> Dual:
        """(u/v)' = (u'v − uv')/v²  - the quotient rule."""
        if not isinstance(other, (Dual, int, float)):
            return NotImplemented
        
        o = Dual.lift(other)
        value = self.val / o.val
        derivative = (self.der * o.val - self.val * o.der) / (o.val ** 2)
        return Dual(value, derivative)

    def __rtruediv__(self, other: float) -> Dual:
        return Dual.lift(other) / self

    def __pow__(self, exponent: float) -> Dual:
        """Only a constant (float) exponent: (uᵏ)' = k·uᵏ⁻¹·u'."""
        value = self.val ** exponent
        derivative = exponent * (self.val ** (exponent - 1)) * self.der
        return Dual(value, derivative)

    # ---- elementary functions (chain rule: (f∘g)' = f'(g)·g') ------------------
    def exp(self) -> Dual:
        value = math.exp(self.val)
        return Dual(value, value * self.der)

    def log(self) -> Dual:
        return Dual(math.log(self.val), self.der / self.val)

    def sin(self) -> Dual:
        return Dual(math.sin(self.val), math.cos(self.val) * self.der)
    
    def cos(self) -> Dual:
        return Dual(math.cos(self.val), -math.sin(self.val) * self.der)


def derivative(f: Callable[[Dual], Dual | float], x: float) -> float:
    """Return f'(x) with a single forward sweep.

    Hint: evaluate f at the dual number x + 1·ε. Beware, f may return a plain
    float (a constant function): its derivative is then 0.
    """
    result = f(Dual(x, 1.0))

    if isinstance(result, Dual):
        return result.der

    return 0.0
