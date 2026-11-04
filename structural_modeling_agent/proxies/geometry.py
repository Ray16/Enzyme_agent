"""Shared geometry helpers owned by the active structural-modeling package."""
import numpy as np


def angle_deg(a, b, c):
    """Angle a--b--c at b; reject undefined coincident-atom geometry."""
    v1, v2 = a-b, c-b
    denominator = np.linalg.norm(v1)*np.linalg.norm(v2)
    if denominator < 1e-12:
        raise ValueError("attack angle is undefined for coincident atoms")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/denominator, -1., 1.))))
