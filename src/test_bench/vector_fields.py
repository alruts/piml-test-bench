from typing import Optional

from jax import numpy as jnp

from test_bench.boundaries import DirichletBC, ImpedanceBC, NeumannBC, PeriodicBC
from test_bench.discretize import SpatialDiscretisation


def laplacian_with_bcs(
    y: SpatialDiscretisation,
    left_bc,
    right_bc,
    v: Optional[SpatialDiscretisation] = None,
):
    """
    y: SpatialDiscretisation of the field
    left_bc, right_bc: BC objects (DirichletBC, NeumannBC, PeriodicBC, ImpedanceBC)
    v: velocity field (needed for ImpedanceBC)
    """
    y_next = jnp.roll(y.vals, shift=-1)
    y_prev = jnp.roll(y.vals, shift=1)
    Δy = (y_next - 2 * y.vals + y_prev) / (y.dx**2)
    Δy = SpatialDiscretisation(y.x0, y.x_final, Δy)

    match left_bc:
        case DirichletBC():
            Δy = left_bc.apply(Δy, left=True)
        case NeumannBC():
            Δy = left_bc.apply(Δy, left=True)
        case PeriodicBC():
            Δy = left_bc.apply(Δy, left=True)
        case ImpedanceBC():
            assert v is not None, "v must not be none for ImpedanceBC"
            Δy, left_bc = left_bc.apply(Δy, v, left=True)
        case _:
            raise ValueError(f"Unknown BC type: {type(left_bc)}")

    match right_bc:
        case DirichletBC():
            Δy = right_bc.apply(Δy, left=False)
        case NeumannBC():
            Δy = right_bc.apply(Δy, left=False)
        case PeriodicBC():
            Δy = right_bc.apply(Δy, left=False)
        case ImpedanceBC():
            assert v is not None, "v must not be none for ImpedanceBC"
            Δy, right_bc = right_bc.apply(Δy, v, left=False)
        case _:
            raise ValueError(f"Unknown BC type: {type(right_bc)}")

    return Δy, left_bc, right_bc


def wave_vector_field(t, state, args):
    """
    t: time
    state: tuple of (p, v) where p, v are SpatialDiscretisation objects
    args: tuple of (c, left_bc, right_bc, v_field)
    """
    (p, v_field), left_bc, right_bc = state  # p = pressure, v = velocity field
    c, _, v_for_impedance = args  # unpack constants & BCs

    dp_dt = v_field  # time derivative of pressure is velocity

    # Compute Laplacian with BCs
    lap, left_bc, right_bc = laplacian_with_bcs(
        p, left_bc=left_bc, right_bc=right_bc, v=v_for_impedance
    )

    dv_dt = SpatialDiscretisation(p.x0, p.x_final, c**2 * lap.vals)

    return (dp_dt, dv_dt), left_bc, right_bc


def gradient(y: SpatialDiscretisation) -> SpatialDiscretisation:
    vals = y.vals
    dx = y.dx

    grad = jnp.empty_like(vals)
    grad = grad.at[1:-1].set((vals[2:] - vals[:-2]) / (2 * dx))
    grad = grad.at[0].set((-3 * vals[0] + 4 * vals[1] - vals[2]) / (2 * dx))
    grad = grad.at[-1].set((3 * vals[-1] - 4 * vals[-2] + vals[-3]) / (2 * dx))

    return SpatialDiscretisation(y.x0, y.x_final, grad)


# def acoustic_vector_field(t, state, args):
#     p, v = state
#     c, ρ = args  # unpack physical constants
#
#     px, vx = enforce_dirichlet_bc((gradient(p), gradient(v)))
#
#     dp_dt = -ρ * c**2 * vx
#     dv_dt = -(1 / ρ) * px
#
#     return (dp_dt, dv_dt)
