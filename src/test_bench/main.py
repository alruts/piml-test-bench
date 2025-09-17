import diffrax
import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from test_bench.discretize import SpatialDiscretisation

jax.config.update("jax_enable_x64", True)


def laplacian(y: SpatialDiscretisation) -> SpatialDiscretisation:
    y_next = jnp.roll(y.vals, shift=1)
    y_prev = jnp.roll(y.vals, shift=-1)
    Δy = (y_next - 2 * y.vals + y_prev) / (y.δx**2)
    # Dirichlet BCs
    Δy = Δy.at[0].set(0)
    Δy = Δy.at[-1].set(0)
    return SpatialDiscretisation(y.x0, y.x_final, Δy)


def wave_vector_field(t, state, args):
    p, v = state  # p = pressure, v = velocity
    c = args  # wave speed
    du_dt = v
    dv_dt = c**2 * laplacian(p)
    return (du_dt, dv_dt)


def gradient(y: SpatialDiscretisation) -> SpatialDiscretisation:
    vals = y.vals
    δx = y.δx

    grad = jnp.empty_like(vals)

    # Interior points: central difference
    grad = grad.at[1:-1].set((vals[2:] - vals[:-2]) / (2 * δx))

    # Boundaries: one-sided differences
    grad = grad.at[0].set((vals[1] - vals[0]) / δx)  # forward difference
    grad = grad.at[-1].set((vals[-1] - vals[-2]) / δx)  # backward difference

    return SpatialDiscretisation(y.x0, y.x_final, grad)


def enforce_dirichlet_bc(state):
    # unpack
    p, v = state
    p_val, v_val = p.vals, v.vals

    # enforce
    p_val = p_val.at[0].set(0)
    p_val = p_val.at[-1].set(0)

    p_updated = SpatialDiscretisation(p.x0, p.x_final, p_val)
    v_updated = SpatialDiscretisation(v.x0, v.x_final, v_val)
    return p_updated, v_updated


def acoustic_vector_field(t, state, args):
    p, v = enforce_dirichlet_bc(state)
    c, ρ = args  # unpack physical constants

    dp_dt = -ρ * c**2 * gradient(v)
    dv_dt = -(1 / ρ) * gradient(p)

    return (dp_dt, dv_dt)


# Parameters
wave_speed = 343.0  # wave speed
density = 1.2  # density
x0 = 0.0
x_final = 1.0
n_points = 800

L = x_final - x0
δx = L / n_points

# Initial conditions
p0_fn = lambda x: jnp.exp(-1000 * (x - x_final / 2) ** 2)  # Gaussian pulse
v0_fn = lambda _: 0.0  # Initially at rest

p0 = SpatialDiscretisation.discretise_fn(x0, x_final, n_points, p0_fn)
v0 = SpatialDiscretisation.discretise_fn(x0, x_final, n_points, v0_fn)

y0 = (p0, v0)

# Time settings
t0 = 0.0
t_final = 0.2
δt = 1e-8
ts = jnp.linspace(t0, t_final, 200)
saveat = diffrax.SaveAt(ts=ts)

# Solver config
term = diffrax.ODETerm(acoustic_vector_field)
solver = diffrax.Tsit5()
rtol = 1e-9
atol = 1e-9
stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

# Solve
sol = diffrax.diffeqsolve(
    term,
    solver,
    t0,
    t_final,
    δt,
    y0,
    args=(wave_speed, density),
    saveat=saveat,
    stepsize_controller=stepsize_controller,
    progress_meter=diffrax.TqdmProgressMeter(),
    max_steps=None,
)

assert sol.ys is not None, "Solution not found"

p_sol, v_sol = sol.ys

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
