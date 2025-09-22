from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


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
    vals: Float[Array, "..."]  # Arbitrary N-dimensional array

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


# Use a more interesting function
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# f(x, y, z) = sin(pi*x) * cos(pi*y) * sin(pi*z)
fn = (
    lambda coord: jnp.sin(jnp.pi * coord[0])
    * jnp.cos(jnp.pi * coord[1])
    * jnp.sin(jnp.pi * coord[2])
)

bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
n_points = [30, 30, 30]  # higher resolution

grid = SpatialDiscretisationND.discretise_fn(bounds, n_points, fn)

print("Shape:", grid.shape)
print("dxs:", grid.dxs)

# Get coordinate arrays for plotting
X, Y, Z = grid.coordinate_arrays  # Each is shape (nx, ny, nz)
vals = grid.vals  # Shape (nx, ny, nz)

# -------------------------------
# 🔹 Option 1: Slice through the middle of Z axis
# -------------------------------
mid_z_idx = n_points[2] // 2

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

x_slice = X[:, :, mid_z_idx]
y_slice = Y[:, :, mid_z_idx]
val_slice = vals[:, :, mid_z_idx]

surf = ax.plot_surface(x_slice, y_slice, val_slice, cmap="viridis")
ax.set_title(f"Slice at z = {Z[0, 0, mid_z_idx]:.2f}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x,y,z)")
fig.colorbar(surf, ax=ax, shrink=0.5)
plt.show()

# -------------------------------
# 🔹 Option 2: 3D scatter plot of sampled points (sparser, slower)
# -------------------------------
# Optional: use only a few points to speed up scatter plot
skip = 3
Xs = X[::skip, ::skip, ::skip].flatten()
Ys = Y[::skip, ::skip, ::skip].flatten()
Zs = Z[::skip, ::skip, ::skip].flatten()
Vs = vals[::skip, ::skip, ::skip].flatten()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
p = ax.scatter(Xs, Ys, Zs, c=Vs, cmap="plasma", s=20)
ax.set_title("3D Scatter of f(x,y,z)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
fig.colorbar(p, ax=ax)
plt.show()
