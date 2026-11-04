"""Separate continuous alignment and proximity observables for pose ensembles."""
import numpy as np


def proximity_weighted_angle_and_distance(poses):
    """Return weighted angle (degrees) and harmonic distance (angstroms).

    The final axis is (distance, angle); the preceding axis enumerates poses.
    These observables are not a combined score, activity or probability.
    """
    x = np.asarray(poses, dtype=float)
    if x.ndim < 2 or x.shape[-1] != 2 or x.shape[-2] == 0:
        raise ValueError('Expected a nonempty pose axis and a distance/angle axis')
    if np.any(~np.isfinite(x)) or np.any(x[..., 0] <= 0):
        raise ValueError('Expected finite positive distances and finite angles')
    if np.any((x[..., 1] < 0) | (x[..., 1] > 180)):
        raise ValueError('Angles must be in degrees between 0 and 180')
    d, a = x[..., 0], x[..., 1]
    w = 1 / d
    return np.stack([(a*w).sum(axis=-1)/w.sum(axis=-1),
                     d.shape[-1]/w.sum(axis=-1)], axis=-1)
