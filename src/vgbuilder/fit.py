"""Fit the procedural body to a sketch by maximising silhouette IoU. Pure numpy, no Blender.

Coordinate descent over the visible parameters with a shrinking step. A candidate costs a few
milliseconds (build + rasterise), so a full fit takes seconds and the Blender build runs once
with the winning parameters. `torso_depth` is not observable from a front view, so it is only
free when a side sketch is given; with a side sketch the objective is a 60/40 blend of the front and
side IoU, which is what separates a thick limb from a long one.
"""
from __future__ import annotations

import numpy as np

from .bodygen import build_body, silhouette, silhouette_side
from .sketch import silhouette_iou
from .spec import DEFAULT_BODY_PARAMS

BOUNDS: dict[str, tuple[float, float]] = {
    "head_scale": (0.6, 1.8),
    "shoulder_half": (0.07, 0.20),
    "waist_half": (0.05, 0.16),
    "hip_half": (0.06, 0.18),
    "leg_frac": (0.35, 0.58),
    "arm_frac": (0.30, 0.48),
    "arm_thickness": (0.5, 2.2),
    "leg_thickness": (0.5, 2.2),
    "arm_angle_deg": (3.0, 60.0),
    "torso_depth": (0.5, 2.2),
}


# The front sketch is the primary one (and has the stricter QA gate), so it counts a bit more.
FRONT_WEIGHT = 0.6


def view_scores(params: dict[str, float], height: float, target: np.ndarray,
                side: np.ndarray | None = None) -> tuple[float, float | None]:
    """(front IoU, side IoU or None) for one candidate body."""
    body = build_body(params, height)
    f = silhouette_iou(silhouette(body), target)
    return f, (silhouette_iou(silhouette_side(body), side) if side is not None else None)


def score(params: dict[str, float], height: float, target: np.ndarray, side: np.ndarray | None = None) -> float:
    """Objective: front IoU, or the 60/40 blend of front and side IoU when a side sketch is given."""
    f, s = view_scores(params, height, target, side)
    return f if s is None else FRONT_WEIGHT * f + (1 - FRONT_WEIGHT) * s


# leg_frac is directly observable (where the legs split), so it is not pulled toward a default.
_PRIOR_KEYS = [k for k in BOUNDS if k != "leg_frac"]


def prior_penalty(params: dict[str, float]) -> float:
    """Sum of squared, range-normalised deviations from anatomical defaults."""
    return sum(((params[k] - DEFAULT_BODY_PARAMS[k]) / (BOUNDS[k][1] - BOUNDS[k][0])) ** 2 for k in _PRIOR_KEYS)


ANCHOR_WEIGHT = 0.5


def anchor_penalty(params: dict[str, float], anchors: dict[str, float]) -> float:
    """Range-normalised squared distance from values read directly off the sketch (e.g. where the legs split).

    Silhouette IoU is almost flat in `leg_frac`, so without this the optimiser can wander to either bound
    for a fraction of a percent of IoU."""
    return ANCHOR_WEIGHT * sum(((params[k] - a) / (BOUNDS[k][1] - BOUNDS[k][0])) ** 2 for k, a in anchors.items())


def optimize(
    params: dict[str, float],
    height: float,
    target: np.ndarray,
    pinned: set[str] | frozenset[str] = frozenset(),
    max_rounds: int = 40,
    prior: float = 0.0,
    side: np.ndarray | None = None,
    anchors: dict[str, float] | None = None,
) -> tuple[dict[str, float], float, int]:
    """Return (best_params, best_iou, evaluations). best_iou is the joint score when `side` is given.

    Maximises IoU - prior * prior_penalty. A silhouette cannot tell a thick limb from a thin one
    with the same outline, so a small prior toward human proportions stops the fit from inflating
    limbs to chase a few pixels. The returned IoU is the plain IoU, without the penalty.
    """
    anchors = {k: v for k, v in (anchors or {}).items() if k in BOUNDS and k not in pinned}
    free = [k for k in BOUNDS if k not in pinned and (side is not None or k != "torso_depth")]
    best = dict(params)
    best_iou = score(best, height, target, side)
    best_s = best_iou - prior * prior_penalty(best) - anchor_penalty(best, anchors)
    evals = 1
    step = 0.14
    for _ in range(max_rounds):
        improved = False
        for k in free:
            lo, hi = BOUNDS[k]
            for sgn in (+1.0, -1.0):
                trial = dict(best)
                trial[k] = float(np.clip(best[k] * (1 + sgn * step), lo, hi))
                if trial[k] == best[k]:
                    continue
                iou = score(trial, height, target, side)
                s = iou - prior * prior_penalty(trial) - anchor_penalty(trial, anchors)
                evals += 1
                if s > best_s + 1e-4:
                    best, best_s, best_iou, improved = trial, s, iou, True
                    break
        if not improved:
            step *= 0.5
            if step < 0.01:
                break
    return best, best_iou, evals
