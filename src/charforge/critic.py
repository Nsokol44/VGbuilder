"""Optional critic hook: an external command (typically a vision-language model wrapper) looks at the
finished asset and returns a score plus notes. charforge does not ship or call any model itself.

Contract
--------
The command is a template; these placeholders are replaced by file paths before it runs
(each argument is substituted separately, so paths with spaces are safe; no shell is involved):

  {contact_sheet}   PNG of the asset (front/side/back/three-quarter renders)
  {silhouette}      front silhouette PNG (characters only; may not exist for effects)
  {context}         JSON file: {"name","kind","style_template","checklist":[...],"metrics":{...}}

It must print one JSON object on stdout (anything before the last `{` line is ignored):

  {"score": 0.0-1.0, "verdict": "pass"|"fail" (optional), "notes": "...", "issues": ["...", ...] (optional)}

A non-zero exit, timeout, unparsable output or an out-of-range score is recorded as a critic *error*,
never as a pass. The critic only annotates the library entry (and flags it below `min_score`);
it never changes the QA status. Verified here only against a stub command, not a real VLM.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .library import Library


def parse_output(text: str) -> dict[str, Any]:
    start = text.rfind("\n{")
    chunk = text[start + 1:] if start >= 0 else text[text.find("{"):] if "{" in text else ""
    try:
        d = json.loads(chunk.strip())
    except (json.JSONDecodeError, ValueError):
        # fall back: whole output may be a (multi-line) JSON object preceded by log noise
        i = text.find("{")
        try:
            d = json.loads(text[i:]) if i >= 0 else None
        except (json.JSONDecodeError, ValueError):
            d = None
    if not isinstance(d, dict):
        raise ValueError("critic did not print a JSON object")
    sc = d.get("score")
    if isinstance(sc, bool) or not isinstance(sc, (int, float)) or not (0.0 <= float(sc) <= 1.0):
        raise ValueError(f"critic score must be a number in [0, 1], got {sc!r}")
    out = {"score": float(sc), "notes": str(d.get("notes", "")), "issues": [str(x) for x in d.get("issues", [])]}
    if d.get("verdict") is not None:
        out["verdict"] = str(d["verdict"])
    return out


def run_critic(lib: Library, name: str, command: str | None = None, min_score: float | None = None,
               timeout_s: int = 300, version: int | None = None) -> dict[str, Any]:
    a = lib.get(name, version)
    if a is None:
        raise KeyError(f"no asset {name!r}")
    crit = a["spec"].get("critic") or {}
    command = command or crit.get("command")
    if not command:
        raise ValueError("no critic command (pass --command or set critic.command in the job)")
    min_score = crit.get("min_score", 0.6) if min_score is None else min_score
    final = lib.resolve(a["final_dir"])
    sheet = final / a["thumb"] if a.get("thumb") else None
    sil = final / a["silhouette"] if a.get("silhouette") else None
    style_checklist: list[str] = []
    if a.get("style_template"):
        from .styles import load_style
        try:
            style_checklist = load_style(a["style_template"]).checklist
        except ValueError:
            pass
    result: dict[str, Any]
    with tempfile.TemporaryDirectory() as td:
        ctx = Path(td) / "context.json"
        ctx.write_text(json.dumps({"name": name, "kind": a["kind"], "style_template": a.get("style_template"),
                                   "checklist": style_checklist, "metrics": a["summary"]}, indent=2))
        subs = {"contact_sheet": str(sheet or ""), "silhouette": str(sil or ""), "context": str(ctx)}
        argv = []
        for tok in shlex.split(command):
            for k, v in subs.items():
                tok = tok.replace("{" + k + "}", v)
            argv.append(tok)
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
            if proc.returncode != 0:
                raise RuntimeError(f"exit {proc.returncode}: {proc.stderr.strip()[-300:]}")
            result = parse_output(proc.stdout)
            result["below_min"] = result["score"] < float(min_score)
            result["min_score"] = float(min_score)
        except Exception as e:  # noqa: BLE001 - any failure is an error record, never a pass
            result = {"score": None, "error": f"{type(e).__name__}: {e}"}
    lib.set_critic(name, result, a["version"])
    return result
