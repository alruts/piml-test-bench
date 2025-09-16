"""
What is needed for a fully functional test bench for 1d optimal sensing problem for impedance

- [x] easy way to specify sensor positions
- [x] easy way to specify impedances
- [ ] source analysis / source correction (would be nice to have (FFT plot))
- feed simulations into a bunch of different algorithms to benchmark (will need different experimental design for each topic)

= current benchmark
    - plane waves
    - DMD?

I need some benchmark simulations to be able to generate clean data. After that
Try to crate a hybrid NODE with a source term / boundary condition term to be able to
inversely infer these things. (is this a way to go? is it mathematically sound?)
(is it possible to embed the 'BCs directly into the ODE' (similar to forcing term))

In any case I will need to make a test bench with 1d and 2d simulations, it would be nice to be able
to make somewhat complex geometry possible in the 2d case ( arc and such ). Possibly this will not work
however, it might.

Todo:
- [ ] Implement sources such as gaussian in 1d and 2d
- [ ] Imple
- [ ]
- [ ]
- [ ]

# current strategy:
    just start with the current (working) 1d example, and try to implement every point ( except geometry ofcourse ) to a working state
    try to see if I want to

    from this I can easily do a sanity check for the hybrid NODE approach, and optimal sensing for PINNs, this might be very good starting point,
    I could then in principle extend to 2 or 3 dimensions in the future (geometry becomes a bit tedious for FD methods).

# braindump:
If I find an efficient differentiable BEM/FEM solver that will be amazing - Comsol might come in clutch.

example file layout
.
├── demo.py
├── pyproject.toml
├── README.md
├── src
│   ├── refrax
│   │   ├── __init__.py
│   │   ├── source.py # implement gaussian source in 1d/2d plus maybe some more complex sources such as harmonic sources and stochastic sources
│   │   ├── sensors.py # compute optimal sensor locations using POD with QR pivoting
│   │   ├── geometry.py # geometry implementation (this will be a headache)
│   │   ├── animate.py # animations in 1d and 2d
│   │   └── impedance.py # vector fitting state-space models to impedance data?
│   └── experiments
│       ├── __init__.py
│       ├── exp1.py
│       └── exp2.py
└── uv.lock

pinn based stuff goes into another library for pinns
uv run
"""

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from diffrax import ODETerm, PIDController, SaveAt, diffeqsolve

jax.config.update("jax_enable_x64", True)

# Parameters
N = int(250) * 8 + 1
L = 1.0
c = 343.0
rho0 = 1.225
x = jnp.linspace(0, L, N, endpoint=False)
dx = L / N
Zc = rho0 * c

# Define zero-pole model for R(s) = k * (s + z) / (s + pole)
k = 0.9
z = 5000.0
pole = 8000.0
A = -pole
B = 1.0
C = k * (z - pole)
D = k

# Initial condition: Gaussian pulse in pressure
p0 = jnp.exp(-1000 * (x - 0.3) ** 2)
v0 = jnp.zeros_like(p0)
U0 = jnp.concatenate(
    [p0, v0, jnp.array([0.0, 0.0])]
)  # Full state vector with filter states for left and right


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


def wave_rhs(_, U, args):
    c, rho0, A, B, C, D, Zc = args
    p = U[0:N]
    v = U[N : 2 * N]
    x_left = U[2 * N]
    x_right = U[2 * N + 1]

    dpdx = central_diff(p, dx)
    dvdx = central_diff(v, dx)

    dpdt = -rho0 * c**2 * dvdx
    dvdt = -(1 / rho0) * dpdx

    # Left boundary with filter
    left = p[0] - Zc * v[0]
    dl_dt_interior = c * dpdx[0] - rho0 * c**2 * dvdx[0]
    dxdt_left = A * x_left + B * left
    dr_dt = C * dxdt_left + D * dl_dt_interior
    dl_dt = dl_dt_interior
    dpdt = dpdt.at[0].set((dr_dt + dl_dt) / 2)
    dvdt = dvdt.at[0].set((dr_dt - dl_dt) / (2 * Zc))

    # Right boundary with filter
    right = p[-1] + Zc * v[-1]
    dr_dt_interior = -c * dpdx[-1] - rho0 * c**2 * dvdx[-1]
    dxdt_right = A * x_right + B * right
    dl_dt = C * dxdt_right + D * dr_dt_interior
    dr_dt = dr_dt_interior
    dpdt = dpdt.at[-1].set((dr_dt + dl_dt) / 2)
    dvdt = dvdt.at[-1].set((dr_dt - dl_dt) / (2 * Zc))

    return jnp.concatenate([dpdt, dvdt, jnp.array([dxdt_left, dxdt_right])])


# Time integration with Diffrax
term = ODETerm(wave_rhs)
solver = diffrax.Dopri8()
t0, t1 = 0.0, 100e-3
dt0 = 1e-12
saveat = SaveAt(ts=jnp.linspace(t0, t1, int(t1 * 32000 * 8)))

# stability check
assert c * (dt0 / dx) <= 1, f"CFL condition is broken got {c * (dt0 / dx)}"

controller = PIDController(rtol=1e-8, atol=1e-12)
sol = diffeqsolve(
    term,
    solver,
    t0=t0,
    t1=t1,
    dt0=dt0,
    y0=U0,
    args=(c, rho0, A, B, C, D, Zc),
    stepsize_controller=controller,
    saveat=saveat,
    max_steps=None,
    progress_meter=diffrax.TqdmProgressMeter(),
)

assert sol.ys is not None, "Solution not obtained"

# Extract and animate pressure
p_time = sol.ys[:, :N]
x_np = jax.device_get(x)
p_np = jax.device_get(p_time)

sensor_positions = [0.54, 0.3, 0.04]
sensor_indices = [jnp.argmin(jnp.abs(x_np - pos)) for pos in sensor_positions]
sensor_waveforms = p_np[:, sensor_indices]

fig, ax = plt.subplots()
(line,) = ax.plot(x_np, p_np[0], label="Pressure")
sensor_dots = ax.plot(
    x_np[sensor_indices],
    p_np[0, sensor_indices],
    "o",
    label="Sensors",
)[0]
ax.set_ylim(-1.0, 1.0)
ax.set_xlabel("x")
ax.set_ylabel("Pressure p(x, t)")
ax.axis("off")
title = ax.set_title("")


def update(frame):
    line.set_ydata(p_np[frame])
    sensor_dots.set_ydata(p_np[frame, sensor_indices])
    title.set_text(f"t = {1e3 * frame * (t1 - t0) / p_np.shape[0]:.3f} ms")
    return line, sensor_dots, title


ani = animation.FuncAnimation(fig, update, frames=len(p_np), blit=False, interval=1)

# Plot sensor waveforms
t_vec = jnp.linspace(t0, t1, len(p_np))
fig, ax = plt.subplots()
for i, idx in enumerate(sensor_indices):
    ax.plot(t_vec * 1e3, sensor_waveforms[:, i], label=f"x = {x_np[idx]:.3f}")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Pressure at sensor")
ax.set_title("Waveforms at Sensor Positions")
ax.legend()
ax.grid(True)

plt.show()
