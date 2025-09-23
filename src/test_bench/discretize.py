from collections.abc import Callable, Sequence
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree


class Point1d(NamedTuple):
    x: float


class Point2d(NamedTuple):
    x: float
    y: float


class Point3d(NamedTuple):
    x: float
    y: float
    z: float


class Vector1d(NamedTuple):
    x: float


class Vector2d(NamedTuple):
    x: float
    y: float


class Vector3d(NamedTuple):
    x: float
    y: float
    z: float


# Type hint unions
Point = Point1d | Point2d | Point3d
Vector = Vector1d | Vector2d | Vector3d
PointOrVector = Point | Vector


class SpatialDiscretisation(eqx.Module):
    x0: float = eqx.field(static=True)
    x_final: float = eqx.field(static=True)
    vals: Float[Array, "n_points"]

    @classmethod
    def discretise_fn(cls, x0: float, x_final: float, n: int, fn: Callable):
        if n < 2:
            raise ValueError("Must discretise [x0, x_final] into at least two points")
        vals = jax.vmap(fn)(jnp.linspace(x0, x_final, n))
        return cls(x0, x_final, vals)

    @property
    def dx(self):
        return (self.x_final - self.x0) / (len(self.vals) - 1)

    @property
    def xs(self):
        return jnp.linspace(self.x0, self.x_final, len(self.vals))

    def binop(self, other, fn):
        if isinstance(other, SpatialDiscretisation):
            if self.x0 != other.x0 or self.x_final != other.x_final:
                raise ValueError("Mismatched spatial discretisations")
            other = other.vals
        return SpatialDiscretisation(self.x0, self.x_final, fn(self.vals, other))

    def __add__(self, other):
        return self.binop(other, lambda x, y: x + y)

    def __mul__(self, other):
        return self.binop(other, lambda x, y: x * y)

    def __radd__(self, other):
        return self.binop(other, lambda x, y: y + x)

    def __rmul__(self, other):
        return self.binop(other, lambda x, y: y * x)

    def __sub__(self, other):
        return self.binop(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self.binop(other, lambda x, y: y - x)


class SpatialDiscretisationND(eqx.Module):
    bounds: Sequence[tuple[float, float]] = eqx.field(static=True)
    vals: Float[Array, "..."]  # N-dimensional array

    @classmethod
    def discretise_fn(
        cls,
        bounds: Sequence[tuple[float, float]],
        n_points: Sequence[int],
        fn: Callable,
    ):
        if len(bounds) != len(n_points):
            raise ValueError("bounds and n_points must have same length")

        if any(n < 2 for n in n_points):
            raise ValueError("Each dimension must have at least 2 points")

        # Generate coordinate arrays per dimension
        axes = [
            jnp.linspace(start, end, num) for (start, end), num in zip(bounds, n_points)
        ]

        # Create meshgrid of coordinates
        mesh = jnp.meshgrid(
            *axes, indexing="ij"
        )  # List of N arrays, each of shape n1 x n2 x ...

        # Stack to get array of shape (..., N), then reshape to (-1, N)
        coords = jnp.stack(mesh, axis=-1).reshape(
            -1, len(bounds)
        )  # shape (prod(n_points), ndim)

        # Evaluate the function on all points
        # fn should accept a vector input of shape (ndim,)
        vals_flat = jax.vmap(fn)(coords)  # shape (prod(n_points),)
        vals = vals_flat.reshape(*n_points)  # Reshape back to grid shape

        return cls(bounds, vals)

    @property
    def shape(self):
        return self.vals.shape

    @property
    def ndim(self):
        return len(self.bounds)

    @property
    def dxs(self):
        return jnp.array(
            [
                (end - start) / (n - 1)
                for (start, end), n in zip(self.bounds, self.vals.shape)
            ]
        )

    @property
    def coordinate_arrays(self):
        axes = [
            jnp.linspace(start, end, num)
            for (start, end), num in zip(self.bounds, self.vals.shape)
        ]
        return jnp.meshgrid(*axes, indexing="ij")  # Returns list of N arrays

    def closest_point_index(self, point: Point):
        """
        Find the closest grid point index to the given Point or Vector.

        Returns:
            A tuple of indices (i, j, ..., n) into the grid.
        """
        # Convert named tuple to jnp array
        point_coords = jnp.array(tuple(point))

        # Get meshgrid of coordinates and stack to shape (..., ndim)
        coord_arrays = (
            self.coordinate_arrays
        )  # list of ndim arrays of shape (n1, n2, ..., nN)
        stacked_coords = jnp.stack(
            coord_arrays, axis=-1
        )  # shape: (n1, n2, ..., nN, ndim)
        flat_coords = stacked_coords.reshape(-1, self.ndim)  # shape: (num_points, ndim)

        # Compute squared Euclidean distances to the input point
        dists_squared = jnp.sum(
            (flat_coords - point_coords) ** 2, axis=1
        )  # shape: (num_points,)

        # Get the index of the closest point
        flat_idx = jnp.argmin(dists_squared)

        # Convert flat index to multi-dimensional index
        multi_idx = jnp.unravel_index(flat_idx, self.vals.shape)

        return multi_idx

    def binop(self, other, fn):
        if isinstance(other, SpatialDiscretisationND):
            if self.bounds != other.bounds or self.vals.shape != other.vals.shape:
                raise ValueError("Mismatched spatial discretisations")
            other = other.vals
        return SpatialDiscretisationND(self.bounds, fn(self.vals, other))

    def __add__(self, other):
        return self.binop(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.binop(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self.binop(other, lambda x, y: y - x)

    def __mul__(self, other):
        return self.binop(other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self.__mul__(other)


class Boundary(eqx.Module):
    """
    Represents a point on a boundary along with its normal vector.

    Attributes:
        apply_fn (Callable): A function ...
        state (PyTree): a PyTree that encodes the current state of the boundary
        point (Point): The position of the boundary point.
        normal (Vector): The outward (or specified) unit normal at the point.
    """

    apply_fn: Callable
    state: PyTree
    point: Point
    normal: Vector


def gradient(y: SpatialDiscretisationND) -> SpatialDiscretisationND:
    grads = []

    for axis, dx in enumerate(y.dxs):
        g = jnp.empty_like(y.vals)

        # central differences: interior points
        center = tuple(
            slice(1, -1) if ax == axis else slice(None) for ax in range(y.ndim)
        )
        left = tuple(
            slice(0, -2) if ax == axis else slice(None) for ax in range(y.ndim)
        )
        right = tuple(
            slice(2, None) if ax == axis else slice(None) for ax in range(y.ndim)
        )

        g = g.at[center].set((y.vals[right] - y.vals[left]) / (2 * dx))

        # forward difference at lower boundary
        start = tuple(0 if ax == axis else slice(None) for ax in range(y.ndim))
        g = g.at[start].set(
            (
                -3 * jnp.take(y.vals, 0, axis=axis)
                + 4 * jnp.take(y.vals, 1, axis=axis)
                - jnp.take(y.vals, 2, axis=axis)
            )
            / (2 * dx)
        )

        # backward difference at upper boundary
        end = tuple(-1 if ax == axis else slice(None) for ax in range(y.ndim))
        g = g.at[end].set(
            (
                3 * jnp.take(y.vals, -1, axis=axis)
                - 4 * jnp.take(y.vals, -2, axis=axis)
                + jnp.take(y.vals, -3, axis=axis)
            )
            / (2 * dx)
        )

        grads.append(g)

    # Stack partial derivatives along a new axis -> (..., ndim)
    grad_vals = jnp.stack(grads, axis=-1)

    return SpatialDiscretisationND(y.bounds, grad_vals)
