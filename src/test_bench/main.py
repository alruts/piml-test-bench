import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from diffrax import (
    Dopri5,
    ODETerm,
    PIDController,
    SaveAt,
    TqdmProgressMeter,
    diffeqsolve,
)  # noqa: E402

from test_bench.discretize import domain, time_axis  # noqa: E402

c = 343.0
rho0 = 1.2
z0 = c * rho0

# spatial domain
N, dx = (200,), (1e-3,)
x = domain(N=(200,), dx=dx, dtype=jnp.float64)

# temporal domain
t_end = 0.1
t, dt = time_axis(
    dx,
    cfl=0.1,
    sound_speed=c,
    total_time=t_end,
    dtype=jnp.float64,
)


def central_diff(f, dx):
    """Central difference scheme."""
    return (f[2:] - f[:-2]) / (2 * dx)


def vector_field(t, y, args):
    p, v = y
    rho_0, c = args

    # pre-compute terms
    dp_dt = p
    dp_dx = p

    dv_dt = v
    dv_dx = v

    # Equation 1
    eqn_1 = dp_dt + rho_0 * c**2 * dv_dx

    # Equation 2
    eqn_2 = dv_dt + (1 / rho_0) * dp_dx

    return (eqn_1, eqn_2)


term = ODETerm(vector_field)
solver = Dopri5()
saveat = SaveAt(ts=t)
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

sol = diffeqsolve(
    term,
    solver,
    t0=t[0],
    t1=t[-1],
    dt0=1e-9,
    y0=0,
    saveat=saveat,
    stepsize_controller=stepsize_controller,
    max_steps=None,
    progress_meter=TqdmProgressMeter(),
)

print(sol.ts)
print(sol.ys)
