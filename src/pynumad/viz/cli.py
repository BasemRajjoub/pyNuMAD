"""``python -m pynumad.viz`` — build a blade and render every viz output.

Usage
-----
    python -m pynumad.viz <blade.yaml> <output_dir> \\
        [--h ELEMENT_SIZE] [--no-adhesive] \\
        [--html | --overview | --components | --adhesive | --all]

Examples
--------
::

    # everything, default element size 0.45 m
    python -m pynumad.viz IEA-22.yaml viz_out/

    # just the interactive HTML, coarser mesh
    python -m pynumad.viz BAR0.yaml viz_out/ --html --h 0.80

    # per-material PNG panels only
    python -m pynumad.viz IEA-22.yaml comp/ --components
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pynumad.viz",
        description="Render PNG snapshots and an interactive HTML view "
                    "of a pyNuMAD blade.",
    )
    p.add_argument("yaml", help="windIO blade .yaml file")
    p.add_argument("output_dir",
                   help="directory to write outputs into (created if missing)")
    p.add_argument("--h", type=float, default=0.45,
                   help="shell element size in metres (default 0.45)")
    p.add_argument("--no-adhesive", action="store_true",
                   help="skip the 3-D adhesive bondlines")
    # output-mode flags: any combination selects those, default = --all
    mode = p.add_argument_group("output modes (default: all)")
    mode.add_argument("--html", action="store_true",
                      help="write interactive HTML (blade_model.html)")
    mode.add_argument("--overview", action="store_true",
                      help="write full-blade overview PNGs + zoom strip")
    mode.add_argument("--components", action="store_true",
                      help="write per-material PNG panels")
    mode.add_argument("--adhesive", action="store_true",
                      help="write adhesive-only PNG snapshots")
    mode.add_argument("--all", action="store_true",
                      help="write every output mode (default if none "
                           "is specified)")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-file progress lines")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    yaml_path = Path(args.yaml)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default to --all when no mode flag was given.
    any_mode = args.html or args.overview or args.components or args.adhesive
    if args.all or not any_mode:
        do_html = do_overview = do_components = do_adhesive = True
    else:
        do_html, do_overview = args.html, args.overview
        do_components, do_adhesive = args.components, args.adhesive

    include_adhesive = not args.no_adhesive

    # Suppress noisy mesh-quality warnings — they're not actionable from
    # this CLI; users who want them should call pynumad directly.
    warnings.filterwarnings("ignore")
    logging.getLogger("pynumad").setLevel(logging.ERROR)

    # Lazy imports so `--help` works without scipy/numpy install.
    import pynumad as pynu
    from pynumad.mesh_gen.mesh_gen import shell_mesh_general

    if not args.quiet:
        print(f"loading {yaml_path.name} and meshing at h={args.h} m ...",
              file=sys.stderr)
    blade = pynu.Blade()
    blade.read_yaml(str(yaml_path))
    mesh = shell_mesh_general(
        blade, forSolid=False,
        includeAdhesive=include_adhesive, elementSize=args.h,
    )
    if not args.quiet:
        print(f"  nodes: {len(mesh['nodes'])}  elements: {len(mesh['elements'])}",
              file=sys.stderr)
        if include_adhesive:
            adh_n = len(mesh.get("adhesiveNds", []))
            adh_e = len(mesh.get("adhesiveEls", []))
            print(f"  adhesive nodes: {adh_n}  adhesive elements: {adh_e}",
                  file=sys.stderr)

    # Defer plotting-backend imports inside writers — they'll raise a
    # clear message if matplotlib/plotly are missing.
    from pynumad.viz import (
        write_adhesive_snapshots,
        write_blade_html,
        write_component_snapshots,
        write_overview_snapshots,
    )

    written: list[Path] = []
    if do_html:
        path = out_dir / "blade_model.html"
        write_blade_html(mesh, path, include_adhesive=include_adhesive,
                         title=f"{yaml_path.stem} mesh (h={args.h} m)")
        written.append(path)
        if not args.quiet:
            print(f"  wrote {path}", file=sys.stderr)
    if do_overview:
        paths = write_overview_snapshots(mesh, out_dir,
                                          include_adhesive=include_adhesive)
        written.extend(paths)
        if not args.quiet:
            print(f"  wrote {len(paths)} overview snapshots", file=sys.stderr)
    if do_components:
        comp_dir = out_dir / "components"
        paths = write_component_snapshots(mesh, comp_dir)
        written.extend(paths)
        if not args.quiet:
            print(f"  wrote {len(paths)} component snapshots into "
                  f"{comp_dir.name}/", file=sys.stderr)
    if do_adhesive and include_adhesive:
        adh_dir = out_dir / "adhesive"
        paths = write_adhesive_snapshots(mesh, adh_dir)
        written.extend(paths)
        if not args.quiet:
            print(f"  wrote {len(paths)} adhesive snapshots into "
                  f"{adh_dir.name}/", file=sys.stderr)

    if not args.quiet:
        print(f"\nDone — {len(written)} files in {out_dir}/", file=sys.stderr)
    return 0
