from typing import Callable

import diffrax
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, PIDController, SaveAt
from jaxtyping import PyTree

from test_bench.animate import animate_pressure_with_sensors
from test_bench.discretize import SpatialDiscretisation
from test_bench.geometry import Boundary, Vector


jax.config.update("jax_enable_x64", True)

# Parameters
n_points = 250 * 8 + 1
L = 1.0
c = 343.0
rho0 = 1.225
x0 = 0.0
dx = L / n_points
Zc = rho0 * c

# Define zero-pole model for R(s) = k * (s + z) / (s + pole)
k = 0.8
z = 100.0
pole = 9000.0
A = -pole
B = 1.0
C = k * (z - pole)
D = k

# Initial condition
p0_fn = lambda x: jnp.exp(-1000 * (x - L / 2) ** 2)  # Gaussian pulse
v0_fn = lambda _: 0.0  # Initially at rest

p0 = SpatialDiscretisation.discretise_fn(x0, L, n_points, p0_fn)
v0 = SpatialDiscretisation.discretise_fn(x0, L, n_points, v0_fn)
init_state = p0, v0, (0.0, 0.0)


def apply_bc_at(state: PyTree, args: PyTree, idx: int, normal: Vector):
    p, v, dpdx, dvdx, dpdt, dvdt, filter_val = state
    A, B, C, D, Zc = args

    if normal == -1:
        # Left boundary
        incoming = p.vals[idx] - Zc * v.vals[idx]
        dl_dt_interior = c * dpdx.vals[idx] - rho0 * c**2 * dvdx.vals[idx]
        dxdt = A * filter_val + B * incoming
        dr_dt = C * dxdt + D * dl_dt_interior
        dl_dt = dl_dt_interior
    elif normal == 1:
        # Right boundary
        incoming = p.vals[idx] + Zc * v.vals[idx]
        dr_dt_interior = -c * dpdx.vals[idx] - rho0 * c**2 * dvdx.vals[idx]
        dxdt = A * filter_val + B * incoming
        dl_dt = C * dxdt + D * dr_dt_interior
        dr_dt = dr_dt_interior
    else:
        raise ValueError("Normal must be -1 (left) or +1 (right)")

    new_dpdt_vals = dpdt.vals.at[idx].set((dr_dt + dl_dt) / 2)
    new_dvdt_vals = dvdt.vals.at[idx].set((dr_dt - dl_dt) / (2 * Zc))

    new_dpdt = SpatialDiscretisation(dpdt.x0, dpdt.x_final, new_dpdt_vals)
    new_dvdt = SpatialDiscretisation(dvdt.x0, dvdt.x_final, new_dvdt_vals)

    return new_dpdt, new_dvdt, dxdt


def central_diff(f, dx):
    """
    Computes central difference with second-order Neumann boundaries.

    Args:
        f: Array to differentiate
        dx: Grid spacing

    Returns:
        df/dx array of same shape as f
    """
    df = jnp.zeros_like(f)
    df = df.at[1:-1].set((f[2:] - f[:-2]) / (2 * dx))  # Interior: O(dx²)
    df = df.at[0].set((-3 * f[0] + 4 * f[1] - f[2]) / (2 * dx))  # Left: O(dx²)
    df = df.at[-1].set((3 * f[-1] - 4 * f[-2] + f[-3]) / (2 * dx))  # Right: O(dx²)
    return df


def gradient(y: SpatialDiscretisation) -> SpatialDiscretisation:
    grad = jnp.empty_like(y.vals)
    grad = grad.at[1:-1].set((y.vals[2:] - y.vals[:-2]) / (2 * dx))
    grad = grad.at[0].set((-3 * y.vals[0] + 4 * y.vals[1] - y.vals[2]) / (2 * dx))
    grad = grad.at[-1].set((3 * y.vals[-1] - 4 * y.vals[-2] + y.vals[-3]) / (2 * dx))

    return SpatialDiscretisation(y.x0, y.x_final, grad)


def wave_field(t, state, args):
    c, rho0, A, B, C, D = args
    Zc = c * rho0

    p, v, (x_left, x_right) = state

    # approximate spatial derivatives
    dpdx = gradient(p)
    dvdx = gradient(v)

    dpdt = -rho0 * c**2 * dvdx
    dvdt = -(1 / rho0) * dpdx

    # Left boundary (normal = -1)
    dpdt, dvdt, val_left = apply_bc_at(
        (p, v, dpdx, dvdx, dpdt, dvdt, x_left), (A, B, C, D, Zc), 0, normal=-1
    )

    # Right boundary (normal = +1)
    dpdt, dvdt, val_right = apply_bc_at(
        (p, v, dpdx, dvdx, dpdt, dvdt, x_right), (A, B, C, D, Zc), -1, normal=1
    )
    return dpdt, dvdt, (val_left, val_right)


# Time integration with Diffrax
term = ODETerm(wave_field)
solver = diffrax.Dopri8()
dt0 = 1e-12
t0, t1 = 0.0, 100e-3
ts = jnp.linspace(t0, t1, int(t1 * 32000 * 8))
saveat = SaveAt(ts=ts)

# stability check
assert c * (dt0 / dx) <= 1, f"CFL condition is broken got {c * (dt0 / dx)}"

controller = PIDController(rtol=1e-8, atol=1e-12)
sol = diffeqsolve(
    term,
    solver,
    t0=t0,
    t1=t1,
    dt0=dt0,
    y0=init_state,
    args=(c, rho0, A, B, C, D),
    stepsize_controller=controller,
    saveat=saveat,
    max_steps=None,
    progress_meter=diffrax.TqdmProgressMeter(),
)

assert sol.ys is not None, "Solution not obtained"

# Create animations
animate_pressure_with_sensors(sol.ys, p0.xs, ts)
