from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
from jaxtyping import PyTree


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


class Boundary(eqx.Module):
    """
    Represents a point on a boundary along with its normal vector.

    Attributes:
        state (PyTree): The state of the Boundary
        point (Point): The position of the boundary point.
        normal (Vector): The outward (or specified) unit normal at the point.
    """

    state: PyTree
    point: Point
    normal: Vector
