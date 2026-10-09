"""Blender-side stages. Everything in this package imports bpy and runs inside Blender
(either the `bpy` Python module or `blender -b -P`). Keep logic that doesn't need Blender
out of here so it stays unit-testable."""
