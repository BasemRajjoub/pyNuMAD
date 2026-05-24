# pyNuMAD — `pce-patches` fork

Fork of Sandia's
[pyNuMAD](https://github.com/sandialabs/pyNuMAD) (BSD-3) extended with
a **full 3D solid composite blade pipeline** for ANSYS:

- New writer `write_ansys_solid_general` — APDL counterpart of the
  existing Abaqus `write_solid_general` (SOLID185, per-section fiber
  CSYS, adhesive bondline).
- Three-stage industry-standard mesh-quality treatment in
  `solidMeshFromShell` (normal smoothing + adaptive layer-thickness
  clamp + Knupp-style untangling) — drives BAR0 bad-Jacobian count
  from 94 → 0.
- End-to-end validation: 54 unit/cross-val tests + 1 ANSYS R2023
  integration test pass.

See [`docs/dev/TODO_solid_3d_pipeline.md`](docs/dev/TODO_solid_3d_pipeline.md)
for the full pickup-anywhere reference, and
[`CHANGELOG.md`](CHANGELOG.md) for the fork-specific commit log.

For everything else (background, original documentation, examples,
license terms), see upstream:
https://github.com/sandialabs/pyNuMAD
