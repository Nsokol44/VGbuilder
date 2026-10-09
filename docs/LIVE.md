# Live link (Blender add-on)

Two ways to run the same build:

- **Pure python / headless** (`vgbuilder run`, the queue, the farm): unchanged. Nothing here needs a Blender window.
- **Live in Blender**: `vgbuilder watch` rebuilds a job whenever you change it, and the add-on shows the result in your
  open Blender and gives you the job's numbers as sliders.

The two never talk to each other directly. They share a folder and three small JSON files, so either side can be a
script, an editor, or an agent.

```
out/_live/<name>/
  live_overrides.json   the editor writes   {"seq": 7, "overrides": {"structure.floors": 4, "params.leg_frac": 0.5}}
  live.json             the watcher writes  state, build_id, seq_applied, result, files, controls, fast, inbox
  live.heartbeat        touched ~1/s while the watcher lives (the add-on shows "watcher not running" when it is stale)
  published/00007/      immutable copy of build 7 (last 3 kept); live.json points here, so a reader never sees a half-written file
  inbox/                what "Send selection to vgbuilder" writes (a .glb, then its .yaml job)
  manifest.json, final/, iter_*/   the ordinary build output of the latest build
```

## Use it

```
vgbuilder watch jobs/struct_hall.yaml -o out          # terminal 1: build now, rebuild on every change (--fast for previews)
vgbuilder addon zip -O vgbuilder_live.zip              # once: Blender > Preferences > Add-ons > Install from Disk
# in Blender: 3D Viewport sidebar > vgbuilder > Session = out/_live/struct_hall > Connect
```

or `vgbuilder blender jobs/struct_hall.yaml --exe /path/to/blender` (starts the watcher and opens Blender with the add-on
loaded from the source folder and connected), or, inside Blender, set the job file and press *Start watcher*.

What changes the build: saving the job `.yaml` in any editor, a slider, `vgbuilder overrides set JOB path=value`, a change to a
sketch or to a file in the job's `styles/` folder. Several quick changes become one rebuild (a change must hold still
for two polls, 0.25 s each, and edits that arrive during a build trigger one more build after it).

## Overrides

Dotted paths into the job (`structure.floors`, `structure.footprint.0`, `effect.radius`, `wear.rust`, `params.leg_frac`,
`accessory_shape.horn_len`, `height_m`, `seed`, ...). They are validated by the same code as a job file, and they count in
the fingerprint. They are limited to the design knobs (`live.ALLOWED_ROOTS`): never a provider, a provider command, a
critic command, a name or a path, because anything that can write the overrides file could otherwise run programs.

- A bad value (floors 99, a window taller than the storey) does not break anything: `live.json` says `invalid` with the
  validator's message and keeps showing the last good build and its sliders.
- Moving a character body slider writes `params.<name>`, which **pins** that parameter: the sketch fit and the auto-fixer
  stop touching it. Untouched sliders show the value the last build used (the fitted one).
- `vgbuilder overrides show|set|clear JOB` read and write the file; `vgbuilder overrides apply JOB` prints a diff of the
  YAML with the overrides folded in, and `--write` replaces the job (keeping `JOB.bak`) and clears the overrides. The
  YAML is re-dumped, so **comments in the job file are lost**; the result is validated before it is written. The add-on's
  *Copy overrides* button puts the same numbers on the clipboard.
- Live builds go to `out/_live/`, never over `out/<name>/`. With no overrides they are byte-identical to `vgbuilder run`
  (tested). With overrides, the files are byte-identical to what `vgbuilder run` makes from the YAML after `overrides apply`
  (tested for a building).

## Speed

A structure rebuilds in about 6 s, an effect in 2 s. A character with a sketch spent most of its time in the sketch
fit (15 of 22 s for `hero_gear`), so the watcher caches the fit per exact input (sketch pixels, height, budget, pinned
params, accessory triangle counts, the fitter's own source): a slider that does not touch those takes about 7 s. Moving an
accessory size or a wear value does not redo the fit; pinning a body parameter does. `--fast` also builds only LOD0
at 256 px with one QA round and no animations; its files are marked `"fast": true`, the panel says so, and they are not
the real output.

## In Blender

The panel shows the job, its state, build status and failure codes, then the controls (sliders for numbers, drop-downs,
toggles; a `*` marks a value you changed, with a reset button), *Reset all*, *Copy overrides*, and *Send to vgbuilder*.

- Each build replaces the imported objects (collection `vgbuilder:<name>`, an empty `vgbuilder:<name>` as their parent).
  **Anything you edit on the imported objects is lost on the next build.** Press *Keep this version* first: it detaches
  the current objects (renames them "(kept ...)") and the next build imports fresh ones. Move/rotate/scale the empty and
  the placement survives rebuilds. Your selection and active object are restored after each import.
- *All LODs + collision* imports the other levels and the collision proxy too (hidden). Effects load their flipbook and
  contact sheet as images instead of objects.
- A build is not loaded while Blender is in Edit mode (the panel says so) and loads when you leave it.
- *Send to vgbuilder*: exports the selected objects as GLB into `inbox/` and writes a job (`provider: import_glb`,
  `height_m` from the selection's world height). Process it with `vgbuilder queue work --inbox out/_live/<name>/inbox --watch`.
  It is a normal import job: it is rigged, LOD'd and QA'd like any other import.
- Starting the watcher from Blender runs the `vgbuilder` program named in the panel; give a full path if Blender does not
  have it on its PATH.

## What is not covered

- Not tested: the sidebar drawing, mouse interaction, real timer ticks, the `vgbuilder blender` launcher, Blender's
  Extensions platform (a manifest ships, the install path tested is the classic *Install from Disk* of a zip), anything
  on Windows or macOS. The add-on's logic (polling, import and replace, sliders, override writing, send-selection, the
  watcher launcher, install/enable through Blender's add-on manager) is tested against the `bpy` module.
- No building from inside Blender (a "level 3" where the add-on runs the generator in-process): every preview is a
  build by the watcher. There is no hand-edit round trip: changes you make to the imported geometry never go back
  into the job.
- One watcher per job and one add-on per session folder. Two Blender windows on one session were not tried; each keeps its
  own copy of the overrides and writes the whole file, so their sliders would overwrite each other.
