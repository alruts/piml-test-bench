import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def animate_pressure_with_sensors(
    sol, xs, ts, sensor_positions=[0.54, 0.3, 0.04], ylim=(-1.0, 1.0), interval=1
):
    """
    Animate and plot pressure waveforms from a simulation result.

    Parameters:
    - sol: Solution object with .ys[0].vals containing time-dependent pressure data.
    - p0: Spatial grid object with .linspace representing x-values.
    - t0: Start time (float).
    - t1: End time (float).
    - sensor_positions: List of x positions for virtual sensors.
    - ylim: Tuple of (ymin, ymax) for pressure plot.
    - interval: Frame interval for animation in milliseconds.
    """

    # Extract data from JAX to NumPy
    p, _, _ = sol
    x_np = jax.device_get(xs)
    p_np = jax.device_get(p.vals)

    # Find closest indices to the sensor positions
    sensor_indices = [jnp.argmin(jnp.abs(x_np - pos)) for pos in sensor_positions]
    sensor_indices = jnp.array(sensor_indices)
    sensor_waveforms = p_np[:, sensor_indices]

    # Create figure for animation
    fig, ax = plt.subplots()
    (line,) = ax.plot(x_np, p_np[0], label="Pressure")
    sensor_dots = ax.plot(
        x_np[sensor_indices],
        p_np[0, sensor_indices],
        "o",
        label="Sensors",
    )[0]

    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("Pressure p(x, t)")
    ax.axis("off")
    title = ax.set_title("")

    def update(frame):
        line.set_ydata(p_np[frame])
        sensor_dots.set_ydata(p_np[frame, sensor_indices])
        time_ms = 1e3 * frame * (ts[-1] - ts[0]) / p_np.shape[0]
        title.set_text(f"t = {time_ms:.3f} ms")
        return line, sensor_dots, title

    ani = animation.FuncAnimation(
        fig, update, frames=len(p_np), blit=False, interval=interval
    )

    # Plot sensor waveforms
    fig2, ax2 = plt.subplots()
    for i, idx in enumerate(sensor_indices):
        ax2.plot(ts * 1e3, sensor_waveforms[:, i], label=f"x = {x_np[idx]:.3f}")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Pressure at sensor")
    ax2.set_title("Waveforms at Sensor Positions")
    ax2.legend()
    ax2.grid(True)

    plt.show()

    return ani  # You can save the animation if needed from the returned object
