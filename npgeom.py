import numpy as np
from numpy.typing import ArrayLike


def distance(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Calculates the array of distances between two arrays of points
    Point coordinates are taken along the last axis
    >>> distance([[1,2],[3,4]], [[4,6],[9,12]])
    array([ 5., 10.])
    >>> distance([[1,2],[9,18]], [4,6])
    array([ 5., 13.])
    """
    x, y = map(np.asarray, (x, y))
    return np.linalg.vector_norm(x - y, axis=-1)


def rotate_vector_2d(vec: ArrayLike, angle: ArrayLike) -> np.ndarray:
    vec, angle = map(np.asarray, (vec, angle))
    if vec.shape[-1] != 2:
        raise ValueError("2d coordinates should be along the last axis.")
    c = np.cos(angle)
    s = np.sin(angle)
    return np.stack(
        (vec[..., 0] * c - vec[..., 1] * s, vec[..., 0] * s - vec[..., 1] * c), axis=-1
    )


def find_perpendicular_vect_2d(vec: ArrayLike) -> np.ndarray:
    """
    Given vectors, returns perpendicular vectors
    >>> find_perpendicular_vect_2d([[1,0],[1,1]])
    array([[ 0,  1],
           [-1,  1]])
    """
    vec = np.asarray(vec)
    if vec.shape[-1] != 2:
        raise ValueError("2d coordinates should be along the last axis.")
    return np.stack((-vec[..., 1], vec[..., 0]), axis=-1)


def signed_area_2d(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> np.ndarray:
    """
    Returns array of signed areas of the paralellograms
    defined by triplets of 2d points
    x - array of 2d points used for the first point
    y - array of 2d points used for the second point
    z - array of 2d points used for the third point
    >>> signed_area_2d(
    ... [[0,0],[0,0],[0,0]],
    ... [[1,0],[1,1],[1,0]],
    ... [[1,1],[1,0],[2,0]])
    array([ 1, -1,  0])
    """
    x, y, z = map(np.asarray, (x, y, z))
    xy = y - x
    xz = z - x
    return xy[..., 0] * xz[..., 1] - xy[..., 1] * xz[..., 0]


def is_point_out_of_rect(
    point: ArrayLike, rect_p1: ArrayLike, rect_p2: ArrayLike | None = None
) -> np.ndarray:
    """
    Returns array of booleans correspondig to whether point is inside the rect
    defined by rect_p1 and rect_p2 (or origin,if None)
    >>> is_point_out_of_rect([[0,0],[-2,0],[2,2]],[-1,1],[1,-1])
    array([False,  True,  True])
    >>> is_point_out_of_rect([[10,10],[200,500],[-200,200]],[640,480])
    array([False,  True,  True])
    """
    point, rect_p1 = map(np.asarray, (point, rect_p1))
    rect_p2 = np.asarray(rect_p2) if rect_p2 else np.zeros(rect_p1.shape)
    return np.any(
        np.logical_or(
            np.logical_and(point < rect_p1, point < rect_p2),
            np.logical_and(point > rect_p1, point > rect_p2),
        ),
        axis=-1,
    )


def is_segment_intersect_1d(
    x1: ArrayLike, x2: ArrayLike, y1: ArrayLike, y2: ArrayLike
) -> np.ndarray:
    """
    Returns array of boolean values indicating whether
    segment x1-x2 intersects segment y1-y2 on 1d line
    x1, x2, y1 and y2 are the arrays of the corresponding values
    >>> is_segment_intersect_1d(
    ... [1,1],
    ... [2,4],
    ... [3,2],
    ... [4,3])
    array([False,  True])
    """
    x = np.stack((x1, x2), axis=-1)
    x.sort()
    y = np.stack((y1, y2), axis=-1)
    y.sort()
    return np.maximum(x[..., 0], y[..., 0]) <= np.minimum(x[..., 1], y[..., 1])


def is_segment_intersect_2d(
    x1: ArrayLike, x2: ArrayLike, y1: ArrayLike, y2: ArrayLike
) -> np.ndarray:
    """
    Returns array of boolean values indicating whether
    segment x1-x2 intersects segment y1-y2 in 2d
    x1, x2, y1 and y2 are the arrays of the corresponding coordinates,
    where 2d coordinates should be given along last axis
    >>> is_segment_intersect_2d(
    ... [[0,0],[0,0],[0,0],[-1,-1]],
    ... [[1,1],[1,1],[1,0],[0,0]],
    ... [[0,1],[-1,-1],[0,1],[1,1]],
    ... [[1,0],[2,2],[1,1],[2,2]])
    array([ True,  True, False, False])
    """
    o1 = signed_area_2d(x1, x2, y1)
    o2 = signed_area_2d(x1, x2, y2)
    o3 = signed_area_2d(y1, y2, x1)
    o4 = signed_area_2d(y1, y2, x2)
    # Determine signs for tuple orientations
    o = np.sign(np.stack((o1, o2, o3, o4), axis=-1))
    # Either orientation changes
    is_orientation_change = (o[..., 0] != o[..., 1]) & (o[..., 2] != o[..., 3])
    # Or all the points are on the same line and intersect in 1d
    is_overlap = np.all(is_segment_intersect_1d(x1, x2, y1, y2), axis=-1)
    colinear_intersect = (o[..., 0] == 0) & is_overlap
    return is_orientation_change | colinear_intersect


def find_segment_intersect_2d(
    x1: ArrayLike, x2: ArrayLike, y1: ArrayLike, y2: ArrayLike
):
    """
    Assuming segments x1-x2 and y1-y2 intersect, find the intersection point
    >>> find_segment_intersect_2d([1,1],[2,2],[1,2],[2,1])
    array([1.5, 1.5])
    >>> find_segment_intersect_2d(
    ... [[1,1],[0,0],[0,0]],
    ... [[2,2],[2,2],[2,1]],
    ... [[1,2],[0,2],[0,2]],
    ... [[2,1],[2,0],[1,0]])
    array([[1.5, 1.5],
           [1. , 1. ],
           [0.8, 0.4]])
    """
    x1, x2, y1, y2 = map(np.asarray, (x1, x2, y1, y2))
    # Defining vectors x & y
    x, y = x2 - x1, y2 - y1
    # Default return value is middle of the y
    # Will be returned if vectors are on the same line
    intersect = y1 + y / 2
    nx = find_perpendicular_vect_2d(x)
    # Find y projection onto nx (in units of size y)
    py = np.sum(nx * y, axis=-1)
    # If projection is zero, then x & y are on the same line,
    # and we will return default
    # Otherwise find the ratio of x1-y1 projected onto nx vs y projection,
    # Which gives the correct amount of vector y to the intersection point
    not_same_line = py != 0
    intersect[not_same_line] = y1[not_same_line] + (
        (
            y[not_same_line]
            * np.expand_dims(
                np.sum(nx * (x1 - y1), axis=-1)[not_same_line] / py, axis=-1
            )
        )
    )
    return intersect


def project_point_to_line(p: ArrayLike, a: ArrayLike, b: ArrayLike):
    """
    >>> project_point_to_line([2,1],[-1,-1],[1,1])
    array([1.5, 1.5])
    """
    p, a, b = map(np.asarray, (p, a, b))
    ap = p - a
    ab = b - a
    k = np.sum(ap * ab, axis=-1) / np.sum(ab * ab, axis=-1)
    return a + ab * np.expand_dims(k, axis=-1)


def point_to_segment_dist_2d(p: ArrayLike, a: ArrayLike, b: ArrayLike):
    """
    >>> point_to_segment_dist_2d([1,2],[1,0],[3,0])
    array([2.])
    """
    p, a, b = map(np.asarray, (p, a, b))
    ap = p - a
    d = distance(a, b)
    u = (b - a) / d
    k = np.atleast_1d(np.sum(ap * u, axis=-1))
    # return np.min(distance(p, np.stack((a, b), axis=0)), axis=0)
    return np.where(
        (k < 0) | (k > d),
        np.minimum(distance(p, a), distance(p, b)),
        distance(p, a + k[..., np.newaxis] * u),
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
