#!/usr/bin/env python3
"""VGbuilder -> TRELLIS.2 bridge. Run by VGbuilder's `command` provider on YOUR GPU machine.

    trellis2_local.py IMAGE OUT.glb [SEED]

VGbuilder's side of the contract: this script runs with the job file's folder as cwd, must
exit 0, and must leave a binary glTF at OUT. Everything TRELLIS.2-specific lives in
run_trellis2() below, so if your TRELLIS.2 version's API differs, that one function is the
fix (VGbuilder surfaces stderr as the provider error in the manifest).

Two ways to drive a generator:

  1. TRELLIS2_CMD set (any generator CLI you already trust, e.g. Hunyuan3D's):
         TRELLIS2_CMD="python gen.py --image {image} --out {out} --seed {seed}"
     Placeholders {image} {out} {seed} are substituted; the command runs and its exit
     code / output file are the contract. This path is covered by VGbuilder's tests.

  2. TRELLIS2_HOME set to a TRELLIS.2 checkout: run_trellis2() imports the image-to-3D
     pipeline from that checkout, samples with the given seed, and exports a GLB with the
     project's own post-processing utils. Written against the TRELLIS.2 public API shape;
     not covered by VGbuilder's tests (no GPU on the build machine).
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys


def run_via_command(image: str, out: str, seed: str) -> int:
    cmd = os.environ["TRELLIS2_CMD"].format(image=image, out=out, seed=seed)
    print(f"[trellis2_local] TRELLIS2_CMD: {cmd}", file=sys.stderr)
    return subprocess.run(shlex.split(cmd), check=False).returncode


def run_trellis2(image: str, out: str, seed: int) -> None:
    home = os.environ["TRELLIS2_HOME"]
    sys.path.insert(0, home)
    os.chdir(home)  # TRELLIS.2 resolves its config/weights relative to the checkout
    from PIL import Image

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.utils import postprocessing_utils

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    img = Image.open(image)
    outputs = pipeline.run(img, seed=seed)
    mesh = outputs["mesh"][0]
    glb = postprocessing_utils.to_glb(mesh) if hasattr(postprocessing_utils, "to_glb") else None
    if glb is None:
        raise RuntimeError("this TRELLIS.2 version has no postprocessing_utils.to_glb; "
                           "export the sampled mesh to GLB here (see TRELLIS.2 examples)")
    glb.export(out)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    image, out = argv[1], argv[2]
    seed = argv[3] if len(argv) > 3 else "0"
    if not os.path.exists(image):
        print(f"[trellis2_local] input image not found: {image}", file=sys.stderr)
        return 2
    if os.environ.get("TRELLIS2_CMD"):
        return run_via_command(image, out, seed)
    if os.environ.get("TRELLIS2_HOME"):
        run_trellis2(image, out, int(seed))
        return 0 if os.path.exists(out) else 1
    print("[trellis2_local] set TRELLIS2_CMD (any generator CLI) or TRELLIS2_HOME "
          "(a TRELLIS.2 checkout) first — see gpu_handoff/README.md", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
