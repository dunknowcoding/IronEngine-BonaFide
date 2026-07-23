"""Copy a freshly built ``bonafide_native.<pyd|so>`` into site-packages.

Invoked by ``build_native_win.bat`` (and reusable on Linux/macOS). Kept
separate from ``build_native.py`` so the batch script has a single,
no-arguments-guessing install step.

Usage:  python scripts/_install_native.py <build_dir>
"""
from __future__ import annotations

import shutil
import sys
import sysconfig
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: _install_native.py <build_dir>", file=sys.stderr)
        return 2
    build_dir = Path(argv[1])
    candidates = (
        list(build_dir.glob("**/bonafide_native*.pyd"))
        + list(build_dir.glob("**/bonafide_native*.so"))
    )
    if not candidates:
        print(f"ERROR: no bonafide_native.* under {build_dir}", file=sys.stderr)
        return 1
    src = candidates[0]
    dst = Path(sysconfig.get_paths()["purelib"]) / src.name
    shutil.copy2(src, dst)
    print(f"installed {src.name} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
