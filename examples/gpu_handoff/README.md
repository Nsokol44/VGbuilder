# GPU handoff: finishing VGbuilder assets on your own GPU

VGbuilder's pipeline (normalise → rig → LOD → wear → QA → export) runs anywhere.
The one stage that wants a GPU is the *mesh generator*. This kit wires a local
image-to-3D generator into VGbuilder's existing `command` provider slot, so the
same job files that produce mannequins today produce generated meshes on your
machine — with every QA gate still applied.

## What you need

- A machine with an NVIDIA GPU (TRELLIS.2-4B wants roughly 16–24 GB VRAM) — or
  any other generator you like; only the wrapper's env vars change.
- This repo, installed the usual way (`pip install -e .`, plus Blender or the
  `bpy` wheel — see the main README).
- A TRELLIS.2 checkout (https://github.com/microsoft/TRELLIS.2), installed per
  its README, **or** any CLI/API that turns one image into a GLB.

## Step 1 — prove the VGbuilder side works (no GPU, 2 minutes)

    vgbuilder run hero_fake.yaml --out /tmp/gpu_selftest

`hero_fake.yaml` drives the shipped *fake* generator (a box wearing the
sketch's silhouette) through the exact same provider slot. If its manifest says
`passed`, the slot, rig, LOD and QA all work on your machine; only the real
generator itself remains unproven.

## Step 2 — point the wrapper at your generator

`trellis2_local.py` (in this folder) is the bridge VGbuilder calls. Pick ONE:

- **TRELLIS.2 checkout:** `export TRELLIS2_HOME=/path/to/TRELLIS.2`
- **Any other generator CLI** (Hunyuan3D server script, AssetFurnace, a hosted
  API shim you wrote): 
  `export TRELLIS2_CMD="python your_gen.py --image {image} --out {out} --seed {seed}"`

Sanity-check the wrapper alone before involving VGbuilder:

    python trellis2_local.py ../sketches/hero.png /tmp/test.glb 7
    # expect: exit 0 and a real GLB at /tmp/test.glb

If this fails, the error is in the generator/wrapper, not VGbuilder — the
TRELLIS.2-specific code is isolated in `run_trellis2()` in the wrapper, and one
function is the whole fix surface.

## Step 3 — run a real job

    vgbuilder run hero_trellis.yaml --out ./out_gpu

or queue several with the GPU limit so the card is never oversubscribed:

    vgbuilder queue add hero_trellis.yaml stocky_trellis.yaml --db ./gpu.db
    vgbuilder queue work --db ./gpu.db --limit gpu=1

Read the manifest like any other job. `provider_error` failures carry the
generator's stderr. Silhouette/wear/LOD failures mean the generator produced
something off-brief: adjust the sketch or seed and re-run — provider results
are cached by (sketch, seed, command), so retries that change nothing are free.

## Honesty box

- The wrapper's `TRELLIS2_CMD` path is covered by VGbuilder's test suite with a
  stub generator. The `TRELLIS2_HOME` path is written against TRELLIS.2's
  public API shape but was **not** run on the machine that wrote it (no GPU
  there). First real run may need a small API fix in `run_trellis2()` — send
  the manifest's provider_error text back and it gets patched.
- Generated meshes still occasionally have shoulder/elbow/head topology the
  rig dislikes. Those land in `needs_review` with the reason, same as
  everything else; they are minutes of cleanup, not silent corruption.
- A/T-pose input sketches give the cleanest rigs. The shipped sketches are
  front-facing neutrals for this reason.
