from typing import Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from test_bench.discretize import SpatialDiscretisation


class DirichletBC(eqx.Module):
    value: float = 0.0  # fixed boundary value

    def apply(self, y: SpatialDiscretisation, left=True):
        idx = 0 if left else -1
        y_new = y.vals.at[idx].set(self.value)
        return SpatialDiscretisation(y.x0, y.x_final, y_new)


class NeumannBC(eqx.Module):
    derivative: float = 0.0  # fixed derivative

    def apply(self, y: SpatialDiscretisation, left=True):
        dx = y.dx
        if left:
            y_val = y.vals.at[0].set(y.vals[1] - dx * self.derivative)
            return SpatialDiscretisation(y.x0, y.x_final, y_val)
        else:
            y_val = y.vals.at[-1].set(y.vals[-2] + dx * self.derivative)
            return SpatialDiscretisation(y.x0, y.x_final, y_val)


class PeriodicBC(eqx.Module):
    def apply(self, y: SpatialDiscretisation, left=True):
        if left:
            y_val = y.vals.at[0].set(y.vals[-1])
        else:
            y_val = y.vals.at[-1].set(y.vals[0])
        return SpatialDiscretisation(y.x0, y.x_final, y_val)


BoundaryCondition = DirichletBC | NeumannBC | PeriodicBC | ImpedanceBC
