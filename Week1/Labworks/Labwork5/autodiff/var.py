"""Reverse mode (backpropagation) on scalars.

Part 2 of the project. Replace every `raise NotImplementedError` by an implementation.
`__add__` is given as a worked example.
"""
from __future__ import annotations

import math


class Var:
    """A node of the computation graph.

    Each node stores its value, its gradient (the adjoint ∂output/∂node, filled by
    `backward`) and its parents together with the *local* derivative
    ∂node/∂parent.
    """

    def __init__(self, value: float, parents: tuple[tuple[Var, float], ...] = ()) -> None:
        self.value = float(value)
        self.grad = 0.0
        self.parents = parents

    def __repr__(self) -> str:
        return f"Var(value={self.value}, grad={self.grad})"

    @staticmethod
    def lift(x: Var | float) -> Var:
        """Turn a plain number into a constant leaf of the graph."""
        if isinstance(x, Var):
            return x
        return Var(float(x))

    # ---- arithmetic: each operation records (parent, local derivative) ---------
    def __add__(self, other: Var | float) -> Var:
        """Worked example: z = x + y, so ∂z/∂x = 1 and ∂z/∂y = 1."""
        if not isinstance(other, (Var, int, float)):
            return NotImplemented
        o = Var.lift(other)
        return Var(self.value + o.value, ((self, 1.0), (o, 1.0)))

    def __radd__(self, other: float) -> Var:
        return self + other
    
    def __neg__(self) -> Var:
        return Var(-self.value, ((self, -1.0),))
    
    def __sub__(self, other: Var | float) -> Var:
        if not isinstance(other, (Var, int, float)):
            return NotImplemented
        return self + (-Var.lift(other))
    
    def __rsub__(self, other: float) -> Var:
        return Var.lift(other) - self

    def __mul__(self, other: Var | float) -> Var:
        """z = x·y: ∂z/∂x = y and ∂z/∂y = x."""
        if not isinstance(other, (Var, int, float)):
            return NotImplemented
        
        o = Var.lift(other)
        return Var(self.value * o.value, ((self, o.value), (o, self.value)),)

    def __rmul__(self, other: float) -> Var:
        return self * other

    def __truediv__(self, other: Var | float) -> Var:
        if not isinstance(other, (Var, int, float)):
            return NotImplemented

        o = Var.lift(other)
        value = self.value / o.value
        parents = ((self, 1.0 / o.value), (o, -self.value / (o.value ** 2)),)
        return Var(value, parents)

    def __rtruediv__(self, other: float) -> Var:
        return Var.lift(other) / self

    def __pow__(self, exponent: float) -> Var:
        value = self.value ** exponent
        local_derivative = exponent * (self.value ** (exponent - 1))
        return Var(value, ((self, local_derivative),))

    # ---- elementary functions ----------------------------------------------------
    def exp(self) -> Var:
        value = math.exp(self.value)
        return Var(value, ((self, value),))

    def log(self) -> Var:
        return Var(math.log(self.value), ((self, 1.0 / self.value),))

    def sin(self) -> Var:
        return Var(math.sin(self.value), ((self, math.cos(self.value)),))

    def cos(self) -> Var:
        return Var(
            math.cos(self.value),
            ((self, -math.sin(self.value)),),)


    # ---- backpropagation ---------------------------------------------------------
    def topological_order(self) -> list[Var]:
        """All nodes of the graph, each node appearing AFTER all of its parents.

        Hint: a depth-first search. Write it WITHOUT recursion (use an explicit
        stack): a recursive version overflows Python's stack on long graphs.
        Mind nodes reachable by several paths: each must appear only once.
        """
        order: list[Var] = []
        visited: set[Var] = set()
        stack: list[tuple[Var, bool]] = [(self, False)]

        while stack:
            node, expanded = stack.pop()
            
            if expanded:
                order.append(node)
                continue

            if node in visited:
                continue

            visited.add(node)
            stack.append((node, True))
            
            for parent, _ in reversed(node.parents):
                if parent not in visited:
                    stack.append((parent, False))
        return order     

    def backward(self) -> None:
        """Fill `.grad` of every node with ∂self/∂node.

        Hint: seed self.grad = 1, then visit the nodes from the output back to the
        inputs, and push each node's gradient to its parents with the chain rule.
        Remember that a node may be used several times...
        """
        nodes = self.topological_order()

        for node in nodes:
            node.grad = 0.0

        self.grad = 1.0

        for node in reversed(nodes):
            for parent, local_derivative in node.parents:
                parent.grad += local_derivative * node.grad
