# VGbuilder

Sketch + a short YAML spec in — a rigged, LOD'd, textured, QA'd game character, a
building, or a VFX flipbook out. Everything is driven by scripts and headless
Blender (the `bpy` module): the YAML and the sketches are the source of truth,
the `.blend` is a build artifact, and the same job always rebuilds
byte-identical `.glb` files.

Three kinds of job:

| Kind | Input | Output |
|---|---|---|
| `character` | front/side sketches + spec | rigged humanoid with LODs, procedural wear textures, optional accessories and a procedural animation set (idle/walk loops + attack, hit, cast, death one-shots, optional root motion and sprite-sheet export) |
| `structure` | numbers only (`structure:` block) | procedural hall / gothic ruin / tower with real openings, roofs, seeded damage, built-in LODs, lightmap UVs, collision proxy |
| `effect` | numbers only (`effect:` block) | fireball / explosion / shockwave / smoke flipbook + particle preset JSON (no Blender needed) |

Every job ends with a status — `passed`, `needs_review` (a normal outcome: read
`final_failures` in the manifest, edit the YAML, rerun), or `error` — and a
`manifest.json` recording every iteration's parameters, QA metrics, failures,
versions, and file hashes.

## Quickstart

```bash
pip install -e ".[dev,blender]"     # the blender extra is the bpy wheel — use Python 3.13 for bpy 5.2
python examples/make_sketch.py      # synthetic test sketches (front + side)
vgbuilder validate jobs/hero_gear.yaml
vgbuilder run jobs/hero_gear.yaml -o out
vgbuilder run jobs -o out -w 4      # or every job in a folder
```

Per job you get `out/<name>/final/` — a `.glb` per LOD, optional FBX, the
`.blend`, textures, a contact sheet and silhouettes — plus the manifest.

> **Headless Linux note:** the `bpy` wheel needs the system EGL/GL libraries
> (`libEGL.so.1`, `libGL.so.1`, Mesa). On a GPU-less machine or VM, run builds
> with `EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1` and make sure the
> Mesa EGL package version matches your installed Mesa drivers.

## Commands

| Command | What it does |
|---|---|
| `vgbuilder validate JOB...` | check job files against the schema without building |
| `vgbuilder run JOB\|DIR -o OUT [-w N]` | build one job or a folder of them |
| `vgbuilder concepts concept.yaml` | parametric silhouette variants, scored for solidity, limb separation and uniqueness; emits jobs for the diverse winners |
| `vgbuilder farm farm.yaml [--enqueue]` | expand deterministic ranges (hue, wear, proportions, knobs) into variant jobs |
| `vgbuilder queue add/list/work/recover` | persistent priority queue: atomic claims, crash recovery, retries only for errors that can change, per-resource limits (one GPU job at a time) |
| `vgbuilder library list/show/review/ingest/silhouettes` | SQLite asset library, versioned by spec fingerprint, with tags and QA summaries |
| `vgbuilder library serve` | local review web page: contact sheets, approve/reject with a note, queue view (binds to 127.0.0.1, no login — don't expose it) |
| `vgbuilder watch JOB -o OUT` | live link: rebuild whenever the YAML, a sketch, or the live overrides change |
| `vgbuilder overrides show/set/clear/apply JOB` | read/write the slider overrides; `apply` folds them into the YAML (`--write` keeps a `.bak`) |
| `vgbuilder addon zip -O vgbuilder_live.zip` | package the Blender add-on |
| `vgbuilder blender JOB --exe /path/to/blender` | start the watcher and open Blender already connected |

A typical batch: `vgbuilder farm farms/souls.yaml --enqueue`,
`vgbuilder queue work -w 2`, `vgbuilder library serve`, review, then export
the approved silhouettes and point a concept file's `exclude_dir` at them so
the next concepts stay different from what you already have.

## Live in Blender

The headless build stays the source of truth. The live link adds a second way
to run the *same* build while you watch it in a running Blender
(full details: [docs/LIVE.md](docs/LIVE.md)):

```bash
vgbuilder watch jobs/struct_hall.yaml -o out     # terminal 1: build now, rebuild on every change
vgbuilder addon zip -O vgbuilder_live.zip        # install in Blender: Preferences > Add-ons > Install from Disk
# in Blender: 3D Viewport sidebar (N) > vgbuilder > Session = out/_live/struct_hall > Connect
```

The add-on needs nothing but Blender. It reads `live.json`, turns the job's
numbers into sliders, writes `live_overrides.json`, and swaps the imported
objects when a build finishes — keeping your placement of the parent empty and
your selection. *Keep this version* detaches the current build so the next one
imports fresh; *Send to vgbuilder* drops the selected meshes into the session
inbox as an import job. Live builds go to `out/_live/`: with no overrides they
are byte-identical to `vgbuilder run`, and after `overrides apply` they are
byte-identical to a run of the baked YAML.

Rebuild speed: an effect ~2 s, a structure ~6 s, a character ~7 s per change
after the first build (the sketch fit is cached). `--fast` builds LOD0 only at
256 px with one QA round; fast files are labelled `"fast": true` and are never
the real output.

Overrides are validated by the same code as job files and can only touch the
design knobs — never a provider, a command, or a path. A bad value (say,
`structure.floors=99`) is refused and the last good build stays up.

## Generators (TRELLIS / Hunyuan / anything image-to-3D)

vgbuilder ships no model code. You give it a program or a server; it handles
the rest of the line.

- `provider: command` / `trellis2` — your program reads `{image}` (and
  `{side_image}`), writes a GLB to `{out}`, exits 0. Extra placeholders:
  `{seed} {name} {faces} {texture_size}`.
- `provider: http` / `hunyuan` — POSTs JSON `{image: <base64 png>, seed, ...}`
  to `url` + `endpoint`; expects GLB bytes or base64 under `response_key`.
  `${VAR}` in headers comes from the environment and never lands in a file.
- Every result is inspected before use (valid GLB, at least one triangle) and
  cached. Failures are structured and carry `retryable` — timeouts, an
  unreachable server and HTTP 429/5xx are retried by the queue; a bad config
  or a non-GLB file is not.
- Model name/version, command or URL (never header values) and the GLB's
  sha256 are embedded in every exported file. Generated meshes then go through
  the same normalise → decimate → rig → LOD → QA path as procedural ones.
- Examples: `examples/providers/*.yaml`.
- **Running a real generator on your own GPU:** `examples/gpu_handoff/` has a
  ready bridge (`trellis2_local.py`, driven by `TRELLIS2_HOME` or any generator
  CLI via `TRELLIS2_CMD`), a no-GPU self-test job (`hero_fake.yaml`), a real
  job (`hero_trellis.yaml`), and a step-by-step README. The wrapper's
  command-mode is test-covered with a stub; the TRELLIS.2 API path itself is
  written against its public API shape and is the one piece not exercised by
  the test suite (no GPU on the build machine).

Licence notes: TRELLIS.2 is MIT (its rendering dependency nvdiffrast has its
own licence). Tencent's Hunyuan3D 2.x community licences exclude the EU, UK
and South Korea.

## Job files

See `jobs/*.yaml` — `hero_gear.yaml` uses most keys. Anything under `params`
is pinned and never auto-adjusted; `qa` overrides thresholds; unknown keys
are rejected at validate time. Highlights:

- `sketch`, `sketch_side`, `side_facing`, `fit_prior` — joint front+side
  silhouette fit, with a limb-proportion report and a "realistic alternative"
- `style_template`, `style_overrides`, `wear`, `texture` — edge wear, grime,
  rust, rot and blood baked to base-colour + roughness; image-space QA
  auto-corrects wear
- `accessories: [pauldrons, horns, cape]`, `accessory_shape` — the cape is
  tagged `vgbuilder_physics=cloth` with pin weights for the engine
- `animations: [idle, walk]` — procedural loops with loop-closure and
  floor-penetration QA
- `animations` also takes one-shots — `attack_light`, `hit`, `cast` (must end
  back at the rest pose) and `death` (must end down; a capped floor-clamp pass
  keeps the prone body out of the ground and reports its lift). Add
  `root_motion: true` to key forward root travel into `walk` / `attack_light`.
- `sprite_sheet: true` — also render the animations from a fixed front camera
  into a transparent sprite sheet + JSON metadata (frame size, constant pivot,
  per-animation rows/fps/loop). Pre-rendered-from-3D for 2D games, labelled as
  such in the metadata — not hand-drawn pixel art.
- textures default to 1024px (request up to 4096); QA additionally gates mean
  saturation, value range (flat/blank detection) and texel density
  (`qa.texture_min_texel_density`, raise it for hero assets).
- `target_engine: generic/unity/unreal/godot` — bone naming, skin-influence
  limits, FBX axes
- `critic: {command, min_score}` — an external scoring hook (JSON contract in
  `src/vgbuilder/critic.py`); annotates the library only

`docs/GAP_ANALYSIS.md` compares the pipeline against an "A-to-Z agentic asset
pipeline" write-up, row by row.

## Testing

```bash
pytest -q tests/test_live.py tests/test_addon.py   # the live link only (~2 min)
pytest -q                                          # everything: 416 tests, ~10 min, mostly Blender builds
```

## What is verified — and what is not

Verified by the test suite (headless `bpy` 5.2.2, CPU only): determinism and
byte-identity, budgets, height, symmetry, skin weights, per-LOD checks, glTF
re-import of every LOD, predicted vs measured silhouette IoU (front and
side), the look-dev bake and auto-correction loop, accessories, animation
checks, embedded-provenance read-back, effect QA and auto-fix, concept
scoring, library versioning/dedupe/relocation, queue ordering, concurrent
claims, crash recovery, retry rules, inbox detection, farm determinism, the
review UI over HTTP, and the critic contract with stub commands. For the live
link: override validation, the watcher loop (rebuild on YAML/override change,
invalid input keeps the last good build, restart continuity), live ==
`run` == baked-YAML byte identity, the fit cache, and the add-on's logic
through the `bpy` module (connect, import/replace, sliders to overrides,
placement and selection kept, keep-this-version, edit-mode pause, failed-import
reporting, send-selection, the watcher launcher, and install/enable of the zip
through Blender's own add-on manager).

**Not verified:**

- **The Blender GUI itself** — panel layout, slider dragging, real timer
  ticks, the `vgbuilder blender` launcher, and anything on Windows or macOS.
  The logic behind all of it is tested; the pixels are not.
- **Real TRELLIS / Hunyuan servers.** The `hunyuan` HTTP defaults come from a
  description of Hunyuan3D-2.x's API server, not a live server. Contract tests
  use mock servers and stub commands, and the 0.78 / 0.30 silhouette gates for
  generated meshes are not calibrated on real output.
- **Real engines.** Unity/Unreal/Godot presets are reasonable defaults;
  nothing was imported into an actual engine. Bone names are mannequin-style,
  not the full UE5 skeleton.
- **A real vision model as critic.** Stub scripts only.

## Limits you will hit

- The procedural character path is a humanoid mannequin — fine for blockouts,
  stylised NPC stand-ins and as a deterministic test bed; not an art pipeline.
  No Rigify, fingers, face rig, or cloth simulation.
- Silhouettes see outlines, not volume; IoU is coarse (256 px), so margins
  under ~0.02 around a threshold are noise. Textures are procedural wear, not
  painted or generated. VFX are numpy flipbooks, not shaders. Animation is two
  procedural cycles — no retargeting or root motion.
- Structures are kit-built boxes: real openings but no glass or interiors
  beyond floor slabs, damage is random per piece, and the collision proxy is a
  solid box (no walking inside). No sketch/footprint input yet.
- The live link previews builds; it never builds inside Blender, and hand
  edits to imported geometry do not flow back into the job.
- The queue is single-machine (SQLite) by design.
