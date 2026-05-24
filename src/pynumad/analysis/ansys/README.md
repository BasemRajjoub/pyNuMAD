# `pynumad.analysis.ansys`

ANSYS APDL deck emission for blade FEA. The writers consume a
`bladeMesh` dict from `pynumad.mesh_gen` and write Mechanical APDL
input decks (`.mac`).

| file | purpose | key functions | entry point |
|---|---|---|---|
| [`write.py`](write.py) | All deck writers | `write_ansys_shell_model`, `write_ansys_solid_general`, `_write_ansys_adhesive`, `write_ansys_loads`, plus per-analysis-type writers (deflections, fatigue, rupture, linear/nonlinear buckling, local fields, resultants) | Y |
| [`read.py`](read.py) | Parse ANSYS output files | `readANSYSoutputs` | N |
| [`run.py`](run.py) | Subprocess launcher for ansys binary | `runAnsys` | N |
| [`utility.py`](utility.py) | Misc APDL helpers used by `write.py` | `txt2mat`, `getLoadFactorsForElementsWithSameSection`, `getMatrialLayerInfoWithOutGUI` | N |
| [`main_ansys_analysis.py`](main_ansys_analysis.py) | High-level driver that wires writers + solver runs | `main_ansys_analysis` | Y |

## Two writer flavours

| writer | element type | mesh source | use case |
|---|---|---|---|
| `write_ansys_shell_model` | SHELL281 / SHELL181 with `SECDATA` composite layups | `get_shell_mesh()` | mid-fidelity stress analysis, buckling, fatigue |
| `write_ansys_solid_general` | SOLID185 (hex + degenerate wedge), one element per ply | `get_solid_mesh()` | high-fidelity 3D solid analysis, inter-laminar stress |

Both share the orthotropic `MP`/`TB,FCLI` material block, the
per-station fiber coordinate systems, and the
`_write_ansys_adhesive` bondline emission.

## Dependencies on other dirs

- [`pynumad.mesh_gen`](../../mesh_gen/) — supplies the mesh dicts.
- [`pynumad.objects.blade`](../../objects/blade.py) — supplies
  `blade.definition.materials` (read by every writer for `MP` / `TB`
  emission).
- [`pynumad.utils.distributed_loading`](../../utils/distributed_loading.py)
  — `spread_concentrated_loads` for tip-load APDL emission.

## Tests

- [`tests/test_ansys_deck.py`](../../tests/test_ansys_deck.py) — shell-deck regressions.
- [`tests/test_ansys_solid_writer.py`](../../tests/test_ansys_solid_writer.py) — solid-deck unit tests.
- [`tests/test_solid_cross_validation.py`](../../tests/test_solid_cross_validation.py) — Abaqus vs ANSYS deck parity.
- [`tests/test_solid_solver_run.py`](../../tests/test_solid_solver_run.py) — end-to-end ANSYS solve (`@integration @slow`; needs `ansys` key in `software_paths.json`).

## Conventions specific to this dir

- Material IDs follow insertion order in `blade.definition.materials` (`kmp + 1` for the `kmp`-th material).
- Element type IDs: 11 = shell main / solid main; 12 = shell181; 21 = mass21; 31 = SOLID185 adhesive.
- Per-section CSYS IDs start at 100 (`100 + section_index`) in
  the solid writer; per-station IDs start at 1000 in the shell writer.
- The failure-criteria sanitiser `_apdl_finite()` coerces None/NaN/inf
  in YAML to documented defaults so ANSYS R2023's strict parser doesn't
  reject the `TB,FCLI` block.
