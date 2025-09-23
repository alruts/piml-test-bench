import jax.numpy as jnp
import matplotlib.pyplot as plt

from test_bench.discretize import Point3d, SpatialDiscretisationND


# define point-wise function
def fn(point):
    x, y, z = point
    return jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y) * jnp.sin(jnp.pi * z)


# Define bounds and resolution
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
n_points = [30, 30, 30]

data = SpatialDiscretisationND.discretise_fn(bounds, n_points, fn)

print("Shape:", data.shape)
print("dxs:", data.dxs)

# Get coordinate arrays for plotting
X, Y, Z = data.coordinate_arrays  # Each is shape (nx, ny, nz)
vals = data.vals  # Shape (nx, ny, nz)

# Option 1: Slice through the middle of Z axis
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

# Option 2: 3D scatterplot of sampled points (sparser, slower)
data = SpatialDiscretisationND.discretise_fn(bounds, n_points, fn)

# process the data in some way
processing_fn = lambda x: jnp.sin(100 * x)
data = SpatialDiscretisationND(bounds, processing_fn(data.vals))

X, Y, Z = data.coordinate_arrays  # Each is shape (nx, ny, nz)
vals = data.vals  # Shape (nx, ny, nz)

# Get coordinate arrays for plotting
X, Y, Z = data.coordinate_arrays  # Each is shape (nx, ny, nz)
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


# Find the closest grid point to (0, 0, 0)
closest_idx = data.closest_point_index(Point3d(0.2, 0.3, 0.4))

# Get the corresponding coordinates
x_star = X[closest_idx]
y_star = Y[closest_idx]
z_star = Z[closest_idx]
# Add red star at closest point
ax.scatter(
    x_star, y_star, z_star, color="red", marker="*", s=100, label="Closest to (0, 0, 0)"
)


fig.colorbar(p, ax=ax)
plt.show()
