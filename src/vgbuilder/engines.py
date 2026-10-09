"""Per-engine export presets: bone naming, skin-influence limit, FBX axes/scale.

Pure data, no Blender. `target_engine` in the job picks a preset; `naming: auto` (the default) takes
the preset's naming. The numbers are the commonly recommended exporter settings for each engine, but
**none of them has been checked by importing into the real engine** (no Unity/Unreal/Godot here).
Every build records the preset it used in the manifest so a wrong guess is easy to find and fix here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EnginePreset:
    name: str
    naming: str                      # key of rigdefs.NAMING
    max_influences: int              # skin weights per vertex the engine's default pipeline keeps
    fbx: dict[str, Any] = field(default_factory=dict)   # kwargs for bpy.ops.export_scene.fbx
    notes: str = ""
    verified_in_engine: bool = False

    def describe(self) -> dict[str, Any]:
        return {"engine": self.name, "naming": self.naming, "max_influences": self.max_influences,
                "fbx": dict(self.fbx), "notes": self.notes, "verified_in_engine": self.verified_in_engine}


PRESETS: dict[str, EnginePreset] = {
    "generic": EnginePreset(
        "generic", naming="ue", max_influences=4,
        fbx={"axis_forward": "-Z", "axis_up": "Y"},
        notes="Mannequin-style names (default before engine presets existed); FBX settings are Blender's defaults.",
    ),
    "unity": EnginePreset(
        "unity", naming="humanoid", max_influences=4,
        fbx={"axis_forward": "-Z", "axis_up": "Y", "apply_scale_options": "FBX_SCALE_ALL"},
        notes="Humanoid bone names (Hips, Spine, LeftUpperArm...) so the Humanoid avatar auto-maps. "
              "4 skin weights is Unity's default quality setting. Prefer glb/fbx import with 'Animation Type: Humanoid'.",
    ),
    "unreal": EnginePreset(
        "unreal", naming="ue", max_influences=8,
        fbx={"axis_forward": "-Y", "axis_up": "Z", "apply_scale_options": "FBX_SCALE_NONE"},
        notes="Mannequin-style names, not the full UE5 skeleton (no clavicles/twist/fingers): retarget with the IK Retargeter.",
    ),
    "godot": EnginePreset(
        "godot", naming="humanoid", max_influences=4,
        fbx={"axis_forward": "-Z", "axis_up": "Y", "apply_scale_options": "FBX_SCALE_ALL"},
        notes="Humanoid names match Godot's humanoid SkeletonProfile. glb is the better import path than fbx.",
    ),
}


def preset(engine: str) -> EnginePreset:
    try:
        return PRESETS[engine]
    except KeyError:
        raise ValueError(f"no preset for engine {engine!r}; known: {sorted(PRESETS)}") from None


def resolve_naming(naming: str, engine: str) -> str:
    return preset(engine).naming if naming == "auto" else naming
