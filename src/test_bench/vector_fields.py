from typing import Optional

from jax import numpy as jnp

from test_bench.boundaries import DirichletBC, ImpedanceBC, NeumannBC, PeriodicBC
from test_bench.discretize import SpatialDiscretisation


def laplacian_with_bcs(
    p: SpatialDiscretisation,
    left_bc,
    right_bc,
    v: Optional[SpatialDiscretisation] = None,
):
    """
    y: SpatialDiscretisation of the field
    left_bc, right_bc: BC objects (DirichletBC, NeumannBC, PeriodicBC, ImpedanceBC)
    v: velocity field (needed for ImpedanceBC)
    """
    pxx = laplacian(p)

    match left_bc:
        case DirichletBC():
            pxx = left_bc.apply(pxx, left=True)
        case NeumannBC():
            pxx = left_bc.apply(pxx, left=True)
        case PeriodicBC():
            pxx = left_bc.apply(pxx, left=True)
        case ImpedanceBC():
            assert v is not None, "v must not be none for ImpedanceBC"
            pxx, left_bc = left_bc.apply(pxx, v, left=True)
        case _:
            raise ValueError(f"Unknown BC type: {type(left_bc)}")

    match right_bc:
        case DirichletBC():
            pxx = right_bc.apply(pxx, left=False)
        case NeumannBC():
            pxx = right_bc.apply(pxx, left=False)
        case PeriodicBC():
            pxx = right_bc.apply(pxx, left=False)
        case ImpedanceBC():
            assert v is not None, "v must not be none for ImpedanceBC"
            pxx, right_bc = right_bc.apply(pxx, v, left=False)
        case _:
            raise ValueError(f"Unknown BC type: {type(right_bc)}")

    return pxx, left_bc, right_bc


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


def laplacian(y: SpatialDiscretisation) -> SpatialDiscretisation:
    y_next = jnp.roll(y.vals, shift=-1)
    y_prev = jnp.roll(y.vals, shift=1)
    yx = (y_next - 2 * y.vals + y_prev) / (y.dx**2)
    return SpatialDiscretisation(y.x0, y.x_final, yx)


def gradient(y: SpatialDiscretisation) -> SpatialDiscretisation:
    grad = jnp.empty_like(y.vals)
    grad = grad.at[1:-1].set((y.vals[2:] - y.vals[:-2]) / (2 * dx))
    grad = grad.at[0].set((-3 * y.vals[0] + 4 * y.vals[1] - y.vals[2]) / (2 * dx))
    grad = grad.at[-1].set((3 * y.vals[-1] - 4 * y.vals[-2] + y.vals[-3]) / (2 * dx))

    return SpatialDiscretisation(y.x0, y.x_final, grad)


def acoustic_vector_field(t, state, args):
    p, v = state
    c, ρ = args  # unpack physical constants

    px, vx = enforce_dirichlet_bc((gradient(p), gradient(v)))

    dp_dt = -ρ * c**2 * vx
    dv_dt = -(1 / ρ) * px

    return (dp_dt, dv_dt)
