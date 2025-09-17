from typing import Optional

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


class ImpedanceBC(eqx.Module):
    b: Float[Array, "n_coeffs"]  # numerator
    a: Float[Array, "n_coeffs"]  # denominator
    v_history: Float[Array, "n_coeffs"]
    p_history: Float[Array, "n_coeffs"]

    def apply(self, y: SpatialDiscretisation, v: SpatialDiscretisation, left=True):
        idx = 0 if left else -1
        v_boundary = v.vals[idx]

        v_hist = jnp.roll(self.v_history, shift=1)
        v_hist = v_hist.at[0].set(v_boundary)

        p_hist = jnp.roll(self.p_history, shift=1)
        p_new = jnp.dot(self.b, v_hist) - jnp.dot(self.a[1:], p_hist[1:])

        # Update histories
        new_bc = ImpedanceBC(self.b, self.a, v_hist, p_hist.at[0].set(p_new))

        # Update boundary value
        y_val = y.vals.at[idx].set(p_new)
        return SpatialDiscretisation(y.x0, y.x_final, y_val), new_bc
