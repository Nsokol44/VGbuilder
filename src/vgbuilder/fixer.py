"""Parameter handling between iterations.

Sketch fitting is now a direct optimisation (vgbuilder.fit) done once, in pure numpy, before
Blender is involved. What is left for the fixer is the mechanical stuff QA can detect after a
real build. Anything not listed here ends up as `needs_review` in the manifest, which is the
point where a human (or an LLM agent reading the contact sheet) edits the job YAML.
"""
from __future__ import annotations

from typing import Any

from . import accessories, fit as fitmod, proportions
from .bodygen import build_body, choose_segments
from .sketch import fit_params as sketch_seed, fit_params_side
from .spec import DEFAULT_BODY_PARAMS, JobSpec


def _extra(spec: JobSpec):
    names = list(spec.accessories)
    return (lambda q: accessories.tri_total(names, q, spec.height_m, spec.accessory_shape)) if names else None


_CODE_FILES = ("fit.py", "sketch.py", "bodygen.py", "proportions.py", "accessories.py", "fixer.py", "rigdefs.py", "spec.py")


def _fit_key(spec: JobSpec, target_mask, side_mask) -> str:
    import hashlib
    import json
    from pathlib import Path
    h = hashlib.sha256()
    for f in _CODE_FILES:                              # a changed fitter must not hit an old answer
        h.update((Path(__file__).parent / f).read_bytes())
    for arr in (target_mask, side_mask):
        h.update(b"-" if arr is None else repr((arr.shape, str(arr.dtype))).encode() + arr.tobytes())
    # The fit only sees accessories through their triangle counts (they take part of the budget), and those do not
    # depend on how big a horn or cape is. So the key holds the counts, not the shape numbers: moving a horn slider
    # does not redo the fit, while anything that would change a count still does.
    acc = ([accessories.tri_total(list(spec.accessories), {**DEFAULT_BODY_PARAMS, "segments": float(sg)}, spec.height_m, spec.accessory_shape)
            for sg in (4, 8, 12, 20)] if spec.accessories else [])
    h.update(json.dumps({"h": spec.height_m, "prior": spec.fit_prior, "budget": spec.resolved_budget(),
                         "seg": spec.resolved_segments(), "params": spec.params, "acc": sorted(spec.accessories),
                         "acc_tris": acc, "facing": spec.side_facing}, sort_keys=True, default=str).encode())
    return h.hexdigest()[:24]


def initial_params(spec: JobSpec, target_mask, side_mask=None, cache_dir=None) -> tuple[dict[str, float], dict[str, Any]]:
    """Seed from the sketch, optimise against it, pick `segments` to fit the budget.

    `cache_dir` (the live link passes one) remembers the answer per exact input, so dragging a slider that does not
    change the fit inputs does not re-run the optimiser. The answer is the same as without the cache."""
    if spec.kind == "structure":                      # nothing to fit: the numbers are the spec
        return {"detail_level": float(spec.params.get("detail_level", 0))}, {}
    if cache_dir is not None and target_mask is not None:
        import json
        from pathlib import Path
        path = Path(cache_dir) / f"{_fit_key(spec, target_mask, side_mask)}.json"
        if path.exists():
            try:
                d = json.loads(path.read_text())
                return d["params"], {**d["info"], "cached": True}
            except (OSError, ValueError, KeyError):
                pass
        p, info = _initial_params(spec, target_mask, side_mask)
        try:
            blob = json.dumps({"params": p, "info": info})
            if json.loads(blob) == {"params": p, "info": info}:          # only cache what survives a JSON round trip exactly
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(blob)
                tmp.replace(path)
        except (TypeError, OSError):
            pass
        return p, info
    return _initial_params(spec, target_mask, side_mask)


def _initial_params(spec: JobSpec, target_mask, side_mask=None) -> tuple[dict[str, float], dict[str, Any]]:
    p = dict(DEFAULT_BODY_PARAMS)
    info: dict[str, Any] = {}
    pinned = {k for k in spec.params if k in DEFAULT_BODY_PARAMS}
    p.update({k: v for k, v in spec.params.items() if k in DEFAULT_BODY_PARAMS})

    seg_pinned = "segments" in spec.params
    p["segments"] = float(spec.resolved_segments())
    if not seg_pinned:
        p["segments"] = float(choose_segments(p, spec.height_m, spec.resolved_budget(), extra=_extra(spec)))
        info["segments_chosen_for_budget"] = int(p["segments"])

    if target_mask is not None:
        seed = {k: v for k, v in sketch_seed(target_mask).items() if k not in pinned}
        if side_mask is not None:
            seed.update({k: v for k, v in fit_params_side(side_mask).items() if k not in pinned})
        p.update(seed)
        info["seed_from_sketch"] = seed
        anchors = {"leg_frac": seed["leg_frac"]} if "leg_frac" in seed else {}
        if not seg_pinned:  # leg/torso changes can shift the count slightly
            p["segments"] = float(choose_segments(p, spec.height_m, spec.resolved_budget(), extra=_extra(spec)))
        seed_iou = fitmod.score(p, spec.height_m, target_mask, side_mask)
        p, iou, evals = fitmod.optimize(p, spec.height_m, target_mask, pinned=pinned, prior=spec.fit_prior, side=side_mask, anchors=anchors)
        if not seg_pinned:
            # body proportions change the triangle count, so settle segments and refit a couple of times
            for _ in range(3):
                seg = choose_segments(p, spec.height_m, spec.resolved_budget(), extra=_extra(spec))
                if float(seg) == p["segments"]:
                    break
                p["segments"] = float(seg)
                p, iou, e2 = fitmod.optimize(p, spec.height_m, target_mask, pinned=pinned, prior=spec.fit_prior, side=side_mask, anchors=anchors)
                evals += e2
            extra = _extra(spec)
            while build_body(p, spec.height_m).tri_count + (extra(p) if extra else 0) > spec.resolved_budget() and p["segments"] > 4:
                p["segments"] -= 1.0
        front, side_iou = fitmod.view_scores(p, spec.height_m, target_mask, side_mask)
        info["fit"] = {"predicted_iou": round(front, 4), "evaluations": evals, "seed_iou": round(seed_iou, 4)}
        if side_iou is not None:
            info["fit"]["predicted_side_iou"] = round(side_iou, 4)
        info["proportions"] = proportions.report(p, spec.height_m)
        if info["proportions"]["warnings"]:
            info["realistic_alternative"] = _realistic_alternative(spec, p, target_mask, side_mask, pinned, front, side_iou, anchors)
    return p, info


def _realistic_alternative(spec, p, target, side, pinned, front, side_iou, anchors, prior: float = 0.5) -> dict[str, Any]:
    """Refit with a strong anatomical prior so the trade-off is visible: IoU given up vs proportions gained."""
    alt, _, _ = fitmod.optimize(p, spec.height_m, target, pinned=pinned, prior=prior, side=side, anchors=anchors)
    f2, s2 = fitmod.view_scores(alt, spec.height_m, target, side)
    out = {
        "prior": prior,
        "front_iou": round(f2, 4), "front_iou_change": round(f2 - front, 4),
        "params": {k: round(v, 4) for k, v in alt.items() if k != "segments"},
        "proportions": proportions.report(alt, spec.height_m),
    }
    if s2 is not None and side_iou is not None:
        out["side_iou"], out["side_iou_change"] = round(s2, 4), round(s2 - side_iou, 4)
    out["hint"] = "to take it: set fit_prior in the job file, or pin the params you trust"
    return out


def next_params(spec: JobSpec, params: dict[str, float], failures: list[dict[str, Any]]) -> tuple[dict[str, float], list[str]]:
    """Return (new_params, human-readable list of what changed)."""
    new, changes = dict(params), []
    codes = {f["code"] for f in failures}
    if spec.kind == "structure":
        if "over_budget" in codes and "detail_level" not in spec.params:
            lv = int(new.get("detail_level", 0))
            if lv < 2:                                # drop trim first (1), then openings (2)
                new["detail_level"] = float(lv + 1)
                changes.append(f"detail_level {lv} -> {lv + 1}")
    elif "over_budget" in codes and spec.provider == "procedural" and "segments" not in spec.params:
        seg = int(new["segments"])
        if seg > 4:
            new["segments"] = float(seg - 1)
            changes.append(f"segments {seg} -> {seg - 1}")
    for f in failures:
        if f["code"] == "texture_wear_coverage" and f.get("fixable"):
            old = float(new.get("wear_scale", 1.0))
            nxt = float(min(3.0, max(0.2, old * (1.35 if f["direction"] == "more" else 1 / 1.35))))
            if nxt != old:
                new["wear_scale"] = round(nxt, 4)
                changes.append(f"wear_scale {old:g} -> {nxt:.3g} (wear {f['direction']})")
            break
    return new, changes
