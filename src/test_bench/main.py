import diffrax
import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from test_bench.boundaries import DirichletBC, ImpedanceBC, NeumannBC
from test_bench.discretize import SpatialDiscretisation
from test_bench.vector_fields import wave_vector_field


jax.config.update("jax_enable_x64", True)


# todo: implement impedances as zero-pole models
# todo: implement sound field sampling
# todo: implement neural pde for discovering the
# todo: implement active sources / not just initial conditions

# Parameters
wave_speed = 343.0  # wave speed
density = 1.2  # density
x0 = 0.0
x_final = 1.0
n_points = 1001

L = x_final - x0
dx = L / n_points

# Initial conditions
p0_fn = lambda x: jnp.exp(-1000 * (x - x_final / 2) ** 2)  # Gaussian pulse
v0_fn = lambda _: 0.0  # Initially at rest

p0 = SpatialDiscretisation.discretise_fn(x0, x_final, n_points, p0_fn)
v0 = SpatialDiscretisation.discretise_fn(x0, x_final, n_points, v0_fn)
y0 = (
    (p0, v0),
    NeumannBC(),
    ImpedanceBC(jnp.array([1.0, 0.0]), jnp.array([0.5]), jnp.array0, 0),
)

# Time settings
t0 = 0.0
t_final = 100e-3
dt = 1e-12
ts = jnp.linspace(t0, t_final, 200)
saveat = diffrax.SaveAt(
    ts=jnp.linspace(
        t0,
        t_final,
        int(t_final * 32000 * 2),
    )
)

# Solver config
term = diffrax.ODETerm(wave_vector_field)
solver = diffrax.Dopri5()
rtol = 1e-9
atol = 1e-12
stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

# check CFL condition
assert wave_speed * (dt / dx) <= 1, (
    f"CFL condition is broken got {wave_speed * (dt / dx)}"
)

# Solve
sol = diffrax.diffeqsolve(
    term,
    solver,
    t0,
    t_final,
    dt,
    y0,
    args=(wave_speed, density, 0),
    saveat=saveat,
    stepsize_controller=stepsize_controller,
    progress_meter=diffrax.TqdmProgressMeter(),
    max_steps=None,
)

assert sol.ys is not None, "Solution not found"

(p_sol, v_sol), _, _ = sol.ys

# animate
fig, ax = plt.subplots(figsize=(6, 4))
(line,) = ax.plot([], [], lw=2)

x_vals = jnp.linspace(x0, x_final, n_points)
ax.set_xlim(x0, x_final)
ax.set_ylim(float(jnp.min(p_sol.vals)), float(jnp.max(p_sol.vals)))
ax.set_xlabel("x")
ax.set_ylabel("Pressure (p)")
ax.set_title("1D Wave Equation Animation")
ax.plot(x_vals, p0.vals)


def init():
    line.set_data([], [])
    return (line,)


def update(frame):
    line.set_data(x_vals, p_sol.vals[frame])
    ax.set_title(f"Time step: {frame}")
    return (line,)


ani = animation.FuncAnimation(
    fig,
    update,
    frames=p_sol.vals.shape[0],
    init_func=init,
    blit=True,
    interval=1,  # in milliseconds
)

plt.show()
