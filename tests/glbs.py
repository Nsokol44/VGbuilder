"""Tiny GLB builder shared by tests."""
from __future__ import annotations

import json
import struct


def make_glb(tris: int = 12, doc_patch: dict | None = None) -> bytes:
    """Smallest valid GLB with `tris` triangles (a fan of copies of one triangle)."""
    pos = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 0, 1) * tris
    doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(pos)}],
           "bufferViews": [{"buffer": 0, "byteLength": len(pos)}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3 * tris, "type": "VEC3",
                          "min": [0, 0, 0], "max": [1, 0, 1]}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}], "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}]}
    doc.update(doc_patch or {})
    j = json.dumps(doc).encode()
    j += b" " * (-len(j) % 4)
    pos += b"\0" * (-len(pos) % 4)
    total = 12 + 8 + len(j) + 8 + len(pos)
    return (b"glTF" + struct.pack("<II", 2, total) + struct.pack("<II", len(j), 0x4E4F534A) + j
            + struct.pack("<II", len(pos), 0x004E4942) + pos)
