#!/usr/bin/env python3
"""Build the shareable browser version of the notebook.

    uv run python export_wasm.py            -> dist/  (read-only app)
    uv run python export_wasm.py --edit     -> dist/  (code visible and editable)

Serve the result over HTTP — opening dist/index.html via file:// will not work:

    python -m http.server --directory dist

Why the staging directory: marimo only bundles a local package into
``public/wheels/`` when the package resolves as a *sibling* of the notebook.
Ours lives under ``src/``, so this copies both into a temp directory and exports
from there, leaving the repo layout alone.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent
NOTEBOOK = ROOT / "notebooks" / "box_prompt.py"
PACKAGE = ROOT / "src" / "phd_boxprompt"
DEFAULT_OUTPUT = ROOT / "dist"


def export(output: Path, mode: str) -> None:
    started = time.monotonic()

    with tempfile.TemporaryDirectory() as raw:
        staging = Path(raw)
        shutil.copy2(NOTEBOOK, staging / NOTEBOOK.name)
        shutil.copytree(
            PACKAGE,
            staging / PACKAGE.name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        staged = time.monotonic()

        # Export straight to the final location. Writing into the staging dir
        # and moving afterwards would copy ~700 files twice whenever the temp
        # dir is on another volume, which is most of the cost on macOS.
        #
        # Captured, not streamed: marimo prints a "serve it from <cwd>" hint
        # that would be misleading. Its output is surfaced only on failure.
        finished = subprocess.run(
            [
                "marimo", "export", "html-wasm", NOTEBOOK.name,
                "-o", str(output.resolve()), "--mode", mode, "--force",
            ],
            cwd=staging,
            capture_output=True,
            text=True,
        )
        if finished.returncode != 0:
            sys.stderr.write(finished.stdout + finished.stderr)
            raise SystemExit(f"marimo export failed (exit {finished.returncode})")

        exported = time.monotonic()

    wheels = sorted((output / "public" / "wheels").glob("phd_boxprompt-*.whl"))
    if not wheels:
        raise SystemExit(
            "phd_boxprompt was not bundled — the notebook would fail in the "
            "browser with an ImportError. Check that marimo can see `uv`."
        )
    written = sum(1 for _ in output.rglob("*") if _.is_file())
    print(
        f"bundled {wheels[0].name}\n"
        f"staged in {staged - started:.1f}s, "
        f"exported {written} files in {exported - staged:.1f}s"
    )

    index = output / "index.html"
    if not index.is_file():
        raise SystemExit(f"no index.html in {output}")

    try:
        shown = output.relative_to(Path.cwd())
    except ValueError:
        shown = output

    print(f"\nWrote {shown}/")
    print(f"\n    python -m http.server --directory {shown}\n")
    print("Then open http://localhost:8000 — first load takes 20-60s while")
    print("Pyodide and the wheels install. file:// will not work.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edit",
        action="store_true",
        help="let viewers see and edit the code (default is a read-only app)",
    )
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    export(args.output, "edit" if args.edit else "run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
