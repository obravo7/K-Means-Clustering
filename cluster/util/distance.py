import numpy as np

# ---------------------------------
# utility functions
# ---------------------------------


def __l1_norm(point, center) -> np.ndarray:
    """
    L1-norm, also commonly referred to as the Manhattan distance
    """
    d = []
    for i in range(len(center)):
        d_i = np.abs(point, center[i])
        d.append(d_i)
    return np.array(d)


def __l2_norm(point, center) -> np.ndarray:
    """
    L2-norm, also commonly referred as the Euclidean distance
    """
    d = []
    for i in range(len(center)):
        d_i = np.sqrt(np.sum((abs(point - center[i])) ** 2, axis=0))
        d.append(d_i)
    return np.array(d)


def l2_norm2(point, center, axis=1) -> np.ndarray:
    """numpy l2-norm. Used for debugging."""
    d = np.linalg.norm(point - center, axis=axis)
    return d

# ------------------------------------------------------------------------------------------
# Note: numpy implementation is faster, but the requirements stated not to use libraries, so
# pure python is used instead.
# ------------------------------------------------------------------------------------------


def l2norm(point, center) -> float:
    """calculate euclidean distance between two n-dimensional points"""
    s = 0
    for x, y in zip(point, center):
        s += (x - y) ** 2
    d = s[0] ** (1 / 2)
    return d


def l2norm_vector(point, centers) -> np.ndarray:
    """calculate l2-norm between a point and several cluster centers."""
    l2 = []
    for center in centers:
        center = center.reshape(1, len(center)).T  # (n, 1)
        norm = l2norm(point, center)
        l2.append(norm)
    return np.array(l2)


def l1norm(point, center) -> float:
    """calculate manhattan distance between two n-dimensional points"""
    s = 0
    for x, y in zip(point, center):
        s += abs(x - y)
    return s[0]


def l1norm_vector(point, centers) -> np.ndarray:
    """calculate l1-norm between a point and several cluster centers."""
    l2 = []
    for center in centers:
        center = center.reshape(1, len(center)).T  # (n, 1)
        norm = l1norm(point, center)
        l2.append(norm)
    return np.array(l2)
