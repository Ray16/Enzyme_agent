#!/usr/bin/env python3
"""Non-overlapping point labels for matplotlib scatter plots.

Written because the campaign's two headline figures annotated EVERY point in a single axes and
became unreadable: 48 two-line labels over 48 markers, with enzyme names stacked on top of each
other. The structural fix is faceting (fewer points per axes, short labels); this module is the
remaining defence for points that still land close together inside a facet.

Method: for each label try a ring of candidate offsets, nearest first, and keep the first that
collides with no already-placed label and no marker. Beyond a short offset a thin leader line is
drawn, so a displaced label is never ambiguous about which point it belongs to. If every candidate
collides the label is placed at the outermost ring anyway WITH a leader -- labels are never silently
dropped, because losing a point's identity in a figure is worse than a tight fit.

Deterministic: candidates are tried in a fixed order, so the same input always yields the same
figure. No randomised repulsion, nothing to re-tune between runs.
"""
import numpy as np

# nearest-first ring of candidate offsets, in points. Short ones read as ordinary labels; the
# outer rings get a leader line.
_CANDIDATES = [
    (0, 11), (0, -12), (13, 0), (-13, 0),
    (10, 9), (-10, 9), (10, -10), (-10, -10),
    (0, 22), (0, -23), (24, 0), (-24, 0),
    (19, 17), (-19, 17), (19, -18), (-19, -18),
    (0, 34), (0, -35), (36, 0), (-36, 0),
    (29, 27), (-29, 27), (29, -28), (-29, -28),
    (0, 48), (0, -49), (50, 0), (-50, 0),
]
_LEADER_FROM = 18.0   # offsets longer than this get a leader line


def place_labels(ax, items, *, fontsize=7.6, color="#44474f", marker_pad=7.0,
                 expand=(1.06, 1.18), zorder=6):
    """Annotate `items` = [(x, y, text), ...] (or [(x, y, text, color), ...]) without overlap.

    Returns the list of created Annotation artists.
    """
    fig = ax.figure
    fig.canvas.draw()                      # one draw so the renderer and transforms are valid
    rend = fig.canvas.get_renderer()

    norm = [(it[0], it[1], it[2], (it[3] if len(it) > 3 else color)) for it in items]
    # place the most crowded points first: they have the fewest free slots
    xy = np.array([[x, y] for x, y, _t, _c in norm], float)
    disp = ax.transData.transform(xy) if len(xy) else np.zeros((0, 2))
    if len(disp) > 1:
        d = np.linalg.norm(disp[:, None, :] - disp[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        crowding = -np.sort(d, axis=1)[:, :3].sum(axis=1)   # smaller nn-distance -> placed first
        order = list(np.argsort(-crowding))
    else:
        order = list(range(len(norm)))

    # markers are obstacles too, so a label never sits on a datapoint
    from matplotlib.transforms import Bbox
    blocked = [Bbox.from_bounds(px - marker_pad, py - marker_pad, 2 * marker_pad, 2 * marker_pad)
               for px, py in disp]

    def make(x, y, text, col, dx, dy):
        """Create the annotation; a displaced label carries a leader line to its point."""
        kw = {}
        if (dx * dx + dy * dy) ** 0.5 > _LEADER_FROM:
            kw["arrowprops"] = dict(arrowstyle="-", color=col, lw=0.6, alpha=0.55,
                                    shrinkA=1.5, shrinkB=4.0)
        return ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                           ha="center", va="center", fontsize=fontsize, color=col,
                           zorder=zorder, **kw)

    placed, out = [], []
    for i in order:
        x, y, text, col = norm[i]
        chosen = None
        others = [b for j, b in enumerate(blocked) if j != i]
        for dx, dy in _CANDIDATES:
            t = make(x, y, text, col, dx, dy)
            bb = t.get_window_extent(rend).expanded(*expand)
            if not any(bb.overlaps(b) for b in placed + others):
                chosen = (t, bb)
                break
            t.remove()
        if chosen is None:                 # everything collided: take the outermost, with a leader
            t = make(x, y, text, col, *_CANDIDATES[-1])
            chosen = (t, t.get_window_extent(rend).expanded(*expand))
        placed.append(chosen[1])
        out.append(chosen[0])
    return out
