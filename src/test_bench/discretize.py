from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array, ScalarLike


def domain(
    N: Tuple[int, ...], dx: Tuple[float, ...], dtype=jnp.float32
) -> Tuple[Array, ...]:
    """
    Generate an N-dimensional rectangular grid.

    Parameters:
    - N: Tuple of ints, number of grid points along each axis (e.g., (128, 128) for 2D).
    - dx: Tuple of floats, grid spacing along each axis (e.g., (0.1e-3, 0.1e-3) for 2D).

    Returns:
    - Tuple of jax.numpy.ndarray: Coordinate arrays for each dimension, shaped according to N.

    Examples:
    >>> X, = domain((5,), (1.0,))
    >>> X.shape
    (5,)
    >>> X
    Array([0., 1., 2., 3., 4.], dtype=float32)

    >>> X, Y = domain((2, 3), (1.0, 2.0))
    >>> X.shape
    (2, 3)
    >>> Y.shape
    (2, 3)
    >>> X[:, 0]
    Array([0., 1.], dtype=float32)
    >>> Y[0, :]
    Array([0., 2., 4.], dtype=float32)
    """
    if len(N) != len(dx):
        raise ValueError("N and dx must be tuples of the same length.")

    axes: Tuple[Array, ...] = tuple(
        jnp.linspace(0, (n - 1) * d, n, dtype=dtype) for n, d in zip(N, dx)
    )
    coords = jnp.meshgrid(*axes, indexing="ij")
    return tuple(coords)


def time_axis(
    dx: Tuple[float, ...],
    cfl: float,
    sound_speed: float,
    total_time: float,
    dtype=jnp.float32,
) -> Tuple[jnp.ndarray, ScalarLike]:
    """
    Generate a 1D time array using the CFL condition.

    Parameters:
    - dx: Tuple of floats, spatial grid spacings.
    - cfl: Float, CFL number (typically <= 1).
    - sound_speed: Float, maximum physical wave speed.
    - total_time: Float, total simulation time.

    Returns:
    - t: jnp.ndarray, 1D array of time points satisfying CFL condition.
    - dt: float, time step used.

    Examples:
    >>> dx = (0.1e-3, 0.1e-3)
    >>> t, dt = time_axis(dx, cfl=0.3, sound_speed=343.0, total_time=1e-3)
    >>> t.shape[0]  # number of time steps
    11434
    >>> jnp.round(dt, 10)
    Array(8.75e-08, dtype=float32)
    >>> jnp.round(t[1] - t[0], 10) == jnp.round(dt, 10)
    Array(True, dtype=bool)
    """
    min_dx = min(dx)
    dt = cfl * min_dx / sound_speed
    num_steps = int(jnp.floor(total_time / dt)) + 1
    t = jnp.linspace(0.0, dt * (num_steps - 1), num_steps, dtype=dtype)
    return t, jnp.array(dt, dtype=dtype)
