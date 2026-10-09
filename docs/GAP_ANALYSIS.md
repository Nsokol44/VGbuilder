# Gap analysis: A-to-Z agentic asset pipeline vs. vgbuilder

Status words are literal. **Built + tested** = code exists and the test suite exercises it headlessly.
**Built, unverified** = code exists but the thing that matters (an engine, a GPU, a real model) was not available here.
**Different** = done another way on purpose. **Not built** = not there.

| Stage in the write-up | Status | What exists / what is missing |
|---|---|---|
| Sketch -> proportions | Built + tested | Front **and** side silhouette fit (joint objective, front weighted 0.6), predicted IoU equals the IoU measured on the Blender mesh. Limb-proportion report with warnings and a "realistic alternative" refit that states its IoU cost. Not a learned proportion prior; ranges are hand-set. |
| Image/mesh generation (TRELLIS etc.) | Built, unverified | Named providers `trellis2` (command + model/version/hints + GPU tag), `hunyuan` and `http` (JSON/base64 POST, GLB bytes or base64-in-JSON back, env-var headers), plus `command` and `import_glb`. Structured, retry-aware errors; GLB inspected before use; model, command/URL and GLB hash embedded in every export; cache keyed on request + sketch + seed. Tested against mock servers and stub commands only. No GPU here, so no real TRELLIS/Hunyuan run, and the Hunyuan field names are unverified defaults (overridable in the job). Generated meshes have their own silhouette gate plus a "collapsed" gate; both uncalibrated. |
| Silhouette concepting | Built + tested | `vgbuilder concepts`: parametric variants, scored on solidity, lobes, limb separation, uniqueness (vs. each other and vs. library silhouettes), greedy diverse pick, emits validated jobs. These are *parametric* variants of the humanoid body, not diffusion concepts. |
| Style (souls-like) | Built + tested | Style templates are data: palette, wear amounts, thresholds measured by QA (saturation, value range, coverage, rot/clean fractions, silhouette rules) plus a human checklist. One built-in template (`souls_like`). |
| Wear / material texturing | Built + tested | Curvature-proxy edge wear, lower-body grime, rust, rot, blood baked to base-colour + roughness; image-space QA with automatic `wear_scale` correction. **Different:** no ArmorPaint/Substance, no diffusion texturing, no true curvature bake from a high-poly. Classification has a noise floor (~0.1). |
| Rig | Built + tested | 18-bone humanoid, heat weights with nearest-bone fallback, influence limits per engine. **Not built:** Rigify (needs the add-on and a full metarig), fingers/face. |
| Accessories / physics tags | Built + tested | Pauldrons, horns (joined), cape (separate mesh) tagged `vgbuilder_physics=cloth` with pin weights as a vertex group and exported attribute. **Not built:** a cloth/soft-body simulation, shrinkwrap-fitted armour on imported meshes (procedural body only). |
| Animation | Built + tested | Procedural idle and walk cycles, loop-closure and floor-penetration QA, exported as glTF/FBX actions. **Not built:** retargeting, mocap, root motion, blend trees. |
| LOD + validation | Built + tested | Per-LOD ratio, silhouette (front+side), height, binding, vertex-group, UV checks. |
| Engine export | Built, unverified | generic / unity / unreal / godot presets (naming, influence limits, FBX axes). **Never imported into any of the engines.** |
| Provenance | Built + tested | Manifest per build plus the recipe embedded in each glb (no timestamps, rebuild stays byte-identical). |
| VFX | Built + tested, **different** | Template flipbooks (fireball/explosion/shockwave/smoke) rendered in numpy, with frame QA and an engine-neutral particle preset JSON. **Not** Geometry Nodes, not a real-time shader, no mesh VFX. |
| Asset library | Built + tested | SQLite, versioned by spec fingerprint, tags, QA summary, files with hashes, review state, relocatable. |
| Job queue / farm | Built + tested | Persistent priority queue, atomic claim, crash recovery, retries only when the error can change (provider errors carry `retryable`), resource tags with per-tag limits (one GPU job at a time, CPU work flows around it), watched inbox, parallel workers, variant farm (deterministic ranges over colour/wear/proportions/accessory shape/effect knobs). Single machine: SQLite, not a distributed queue. |
| Review UI | Built + tested (HTTP level) | Local stdlib web UI: browse, view contact sheet, approve/reject with note, queue page. Not tested in a browser; no login (binds to 127.0.0.1). |
| VLM critic | Built, unverified | Command-hook contract (JSON in/out), errors never count as pass, annotates the library only. **Tested only with stub scripts**, never a real vision model. |
| Buildings / environments | Built + tested (single buildings), **different** | `kind: structure`: hall / gothic / tower presets, real openings, roofs, trim, seeded damage, build-time LODs, lightmap UV set, collision proxy, QA against the source numbers, same look-dev/library/queue/farm/review path. **Not** Geometry Nodes (Python-generated geometry). **Not built:** modular-kit placement, environment composition, footprint/massing sketch input, image-to-3D hero props inside structures, impostors/occlusion meshes, moss/orientation wear masks, interiors, structural plausibility of damage. |
| Closed-loop iteration | Built + tested | Mechanical fixers only (budget, wear, VFX radius/fade). Anything subjective ends as `needs_review`. |

## The second review (providers, buildings): what was and was not taken

| Suggestion | Outcome |
|---|---|
| Narrow provider contract with structured failures the queue can retry | Done: error codes + `retryable`, queue obeys it |
| `provider: trellis2 / hunyuan` field, procedural stays the default and the CI path | Done |
| Record model name/version and exact command in provenance | Done (embedded in the glb; header values never recorded) |
| Silhouette overlap as a soft gate on the new mesh | Done as its own QA codes; thresholds are placeholders until real output exists |
| Mark GPU jobs, limit concurrency | Done (resource tags, `--limit`) |
| Mock provider + contract tests; one golden-path GPU test later | Mock servers/stubs done; **the GPU golden-path test does not exist** |
| Remesh/decimate dense generator output | Already existed (budget decimation) |
| AssetFurnace as a backend | Not built; it would be one more `command` wrapper |
| `trellis.cpp` / ComfyUI nodes | Not built; reachable through `command` / `http` if they expose a CLI or server |
| `structure` job type with one solid generator end to end | Done, with Python geometry instead of Geometry Nodes |
| Modular kits, composition, impostors, moss/north-face masks | Not built |

## The live link (run via pure python, and live in Blender)

| Requirement | Outcome |
|---|---|
| Pure-python / headless build keeps working, unchanged | Yes: `vgbuilder run`, queue and farm do not use any of it; live builds go to `out/_live/` |
| See the result in a running Blender | Add-on: auto-reload of the latest build, replace-in-place, placement kept, selection kept |
| Edit parameters live | Sliders / drop-downs / toggles generated from the job (`live.controls`), written as overrides, rebuilt by `vgbuilder watch` |
| Same result either way | Tested: live build == `vgbuilder run` byte for byte; building from slider values == building from the YAML after `overrides apply` |
| Send a Blender model into the pipeline | "Send selection": GLB + import job into the inbox, same QA/LOD/rig path as any import |
| Preview that updates while the slider is held | **Not built**: each change is a build (structure ~6 s, effect ~2 s, sketched character ~7 s after the first) |
| Hand edits in Blender flow back into the job | **Not built** (and not possible for a generated mesh); "Keep this version" detaches objects instead |
| GUI behaviour in a real Blender window | **Not tested** (only the logic through the `bpy` module) |

## What would be the next honest steps
1. Import one exported character and one building into Godot/Unity/Unreal and fix whatever the presets got wrong
   (check the second UV set, the `UCX_` collision naming and the axis conversion).
2. Run a real image-to-3D model through `trellis2` / `hunyuan` on a GPU box, correct the request fields if needed and
   calibrate the generated-mesh silhouette gates on real output.
3. Point the critic hook at a real VLM and measure whether its scores agree with the numeric QA.
4. Rigify / true cloth simulation only if the target engine actually needs them.
