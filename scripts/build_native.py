"""Build the ``bonafide_native`` C++/CUDA extension.

What this script does
---------------------
1. Probes every prerequisite (Python, CUDA toolkit, MSVC/host C++ compiler,
   CMake, nanobind) and prints a short report with a clear PASS / FAIL per
   item.
2. If any prerequisite is missing it exits non-zero with an actionable
   install hint — no half-attempted builds.
3. Otherwise it configures and builds the extension into ``native/build/``
   and copies the resulting ``bonafide_native.<pyd|so>`` next to the
   active interpreter's site-packages so ``import bonafide_native`` works.

This is the recommended way to build the native layer. The CMakeLists is
authoritative; this script is a friendlier wrapper that diagnoses common
pitfalls.

Usage
-----
    # default (Release, auto-detect everything)
    python scripts/build_native.py

    # debug build, custom out dir
    python scripts/build_native.py --debug --build-dir /tmp/bnf-build

    # diagnose only
    python scripts/build_native.py --check
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NATIVE_DIR = REPO_ROOT / "native"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fix: str = ""


# --------------------------------------------------------------- checks
def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def check_python() -> CheckResult:
    v = sys.version_info
    ok = v >= (3, 11)
    return CheckResult(
        "Python ≥ 3.11", ok,
        detail=f"{platform.python_version()} at {sys.executable}",
        fix="Activate the IronEngineWorld conda env (Python 3.11)." if not ok else "",
    )


def check_cmake() -> CheckResult:
    p = _which("cmake")
    if not p:
        return CheckResult("CMake", False, detail="not on PATH",
                           fix="Install CMake ≥ 3.24 from https://cmake.org/download/ "
                               "or `conda install cmake`.")
    out = subprocess.run([p, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    ok = "version 3." in out
    return CheckResult("CMake", ok, detail=f"{out}  ({p})",
                       fix="" if ok else "CMake ≥ 3.24 required.")


def check_cuda() -> CheckResult:
    nvcc = _which("nvcc")
    if not nvcc:
        return CheckResult("CUDA toolkit", False, detail="nvcc not on PATH",
                           fix="Install the NVIDIA CUDA Toolkit "
                               "(https://developer.nvidia.com/cuda-toolkit). "
                               "Add `<install>/bin` to PATH.")
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    ver = ""
    for line in out.splitlines():
        if "release" in line:
            ver = line.strip()
            break
    return CheckResult("CUDA toolkit (nvcc)", True, detail=f"{ver}  ({nvcc})")


def check_host_cxx() -> CheckResult:
    if platform.system() == "Windows":
        cl = _which("cl.exe")
        if cl:
            return CheckResult("Host C++ compiler (cl.exe)", True, detail=cl)

        # Two distinct failure modes worth telling apart:
        #   (a) No VS at all     → run the installer
        #   (b) VS shell present but C++ workload missing → modify the install
        #   (c) VS + workload present, just not on PATH    → activate vcvars
        vs_roots = [
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Community"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Professional"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Enterprise"),
        ]
        vs_root = next((p for p in vs_roots if p.exists()), None)
        if vs_root is None:
            return CheckResult(
                "Host C++ compiler (cl.exe)", False, detail="no VS 2022 install found",
                fix="Install 'Visual Studio Build Tools 2022' with the 'Desktop "
                    "development with C++' workload from "
                    "https://visualstudio.microsoft.com/downloads/. "
                    "After install, run from a 'x64 Native Tools Command Prompt'.",
            )
        cls_under_root = list(vs_root.glob("VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe"))
        if not cls_under_root:
            return CheckResult(
                "Host C++ compiler (cl.exe)", False,
                detail=f"VS at {vs_root} but the C++ workload is not installed",
                fix=(
                    "Open the VS Installer, pick 'Modify', and enable\n"
                    "  - Desktop development with C++   (workload)\n"
                    "  - MSVC v143 - VS 2022 C++ x64/x86 build tools  (component)\n"
                    "  - Windows 10/11 SDK                              (component)\n"
                    f"Or run unattended:\n"
                    f"  C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\setup.exe modify"
                    f" --installPath \"{vs_root}\" --add Microsoft.VisualStudio.Workload.VCTools"
                    f" --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
                    f" --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --quiet"
                ),
            )
        return CheckResult(
            "Host C++ compiler (cl.exe)", False,
            detail=f"VS + C++ workload at {vs_root}; cl.exe found at {cls_under_root[0]}",
            fix=f"Activate vcvars first:\n"
                f"  & \"{vs_root / 'VC/Auxiliary/Build/vcvarsall.bat'}\" x64\n"
                f"or run from a 'x64 Native Tools Command Prompt for VS 2022'.",
        )
    # Linux / macOS
    cc = _which("g++") or _which("clang++")
    if cc:
        return CheckResult("Host C++ compiler", True, detail=cc)
    return CheckResult("Host C++ compiler", False, detail="g++ / clang++ not on PATH",
                       fix="Install build-essential (Linux) or Xcode CLT (macOS).")


def check_cuda_vs_integration() -> CheckResult:
    """On Windows, CMake's CUDA support requires .props files to live under
    every VS install's BuildCustomizations folder. Without them the
    'No CUDA toolset found' error fires."""
    if platform.system() != "Windows":
        return CheckResult("CUDA VS integration", True, detail="(not applicable)")
    src = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if not src.exists():
        return CheckResult("CUDA VS integration", False,
                           detail="CUDA install folder missing",
                           fix="Install the CUDA toolkit first.")
    # Look for any version with the integration files present.
    candidates = sorted(src.glob("v*/extras/visual_studio_integration/MSBuildExtensions"))
    if not candidates:
        return CheckResult(
            "CUDA VS integration", False,
            detail="MSBuildExtensions directory missing under any CUDA install",
            fix="Reinstall CUDA toolkit with the 'Visual Studio Integration' component checked.",
        )
    # Look for at least one VS 2022 install with the .props file copied in.
    vs_roots = [
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2022"),
        Path("C:/Program Files/Microsoft Visual Studio/2022"),
    ]
    target_glob = "*/MSBuild/Microsoft/VC/v170/BuildCustomizations/CUDA*.props"
    for root in vs_roots:
        if root.exists() and any(root.glob(target_glob)):
            return CheckResult("CUDA VS integration", True,
                               detail=f"CUDA props installed under {root}")
    return CheckResult(
        "CUDA VS integration", False,
        detail="Found CUDA installer files but no VS 2022 BuildCustomizations entry",
        fix=(
            f"Copy the four .props/.targets/.xml files from\n"
            f"  {candidates[-1]}\n"
            f"into\n"
            f"  <VS-install>/MSBuild/Microsoft/VC/v170/BuildCustomizations/\n"
            f"(or use the CUDA toolkit installer 'modify' to enable VS integration). "
            f"Skip this step if you build with `-G Ninja`."
        ),
    )


def check_nanobind() -> CheckResult:
    try:
        import nanobind
        cmake_dir = nanobind.cmake_dir()
        return CheckResult("nanobind", True, detail=f"{nanobind.__version__} ({cmake_dir})")
    except ImportError:
        return CheckResult(
            "nanobind", False, detail="not installed",
            fix=f"{sys.executable} -m pip install nanobind",
        )


# --------------------------------------------------------------- build
def configure_and_build(build_dir: Path, debug: bool) -> int:
    config = "Debug" if debug else "Release"
    cmake = _which("cmake") or "cmake"
    args = [
        cmake, "-S", str(NATIVE_DIR), "-B", str(build_dir),
        f"-DCMAKE_BUILD_TYPE={config}",
        f"-DPython_EXECUTABLE={sys.executable}",
    ]
    print(f"\n>>> {' '.join(args)}")
    rc = subprocess.run(args).returncode
    if rc != 0:
        return rc
    args = [cmake, "--build", str(build_dir), "--config", config, "-j"]
    print(f"\n>>> {' '.join(args)}")
    return subprocess.run(args).returncode


def install_extension(build_dir: Path) -> Path | None:
    candidates = (
        list(build_dir.glob("**/bonafide_native*.pyd"))
        + list(build_dir.glob("**/bonafide_native*.so"))
    )
    if not candidates:
        print("ERROR: built extension not found under", build_dir, file=sys.stderr)
        return None
    src = candidates[0]
    site = sysconfig.get_paths()["purelib"]
    dst = Path(site) / src.name
    shutil.copy2(src, dst)
    print(f"copied {src} → {dst}")
    return dst


# --------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="diagnose prereqs only; don't build")
    parser.add_argument("--debug", action="store_true",
                        help="build Debug instead of Release")
    parser.add_argument("--build-dir", type=Path,
                        default=NATIVE_DIR / "build",
                        help="CMake binary dir (default: native/build)")
    args = parser.parse_args(argv)

    print("=" * 64)
    print("bonafide_native build doctor")
    print("=" * 64)
    checks = [
        check_python(),
        check_cmake(),
        check_cuda(),
        check_host_cxx(),
        check_cuda_vs_integration(),
        check_nanobind(),
    ]
    for c in checks:
        mark = "PASS" if c.ok else "FAIL"
        print(f"  [{mark}] {c.name:<30} {c.detail}")
        if not c.ok and c.fix:
            for line in c.fix.splitlines():
                print(f"         -> {line}")

    failures = [c for c in checks if not c.ok]
    if failures:
        print(f"\n{len(failures)} prerequisite(s) missing — fix them and retry.")
        return 1

    if args.check:
        print("\nAll prereqs satisfied. Re-run without --check to build.")
        return 0

    args.build_dir.mkdir(parents=True, exist_ok=True)
    rc = configure_and_build(args.build_dir, debug=args.debug)
    if rc != 0:
        print(f"\nBuild failed with exit code {rc}.", file=sys.stderr)
        return rc

    if not install_extension(args.build_dir):
        return 2

    print("\nBuild complete. Try:  python -c \"import bonafide_native; print('OK')\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
