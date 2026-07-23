"""Probe runtime for required + optional modules. CI / setup helper."""
import sys
import importlib.util


CORE = ["torch", "numpy", "loguru", "rich", "imageio", "PIL", "pygltflib", "pytest"]
OPTIONAL = [
    "pytestqt",
    "wgpu",
    "cupy",
    "gsplat",
    "nvdiffrast",
    "warp",
    "tinycudann",
    "pyktx",
    "openvdb",
    "pxr",
    "PySide6",
    "ironengine_3d_creator",
    "ironengine_sim",
]


def main() -> int:
    print(f"python  {sys.version.split()[0]}  ({sys.executable})")
    print()
    print("CORE")
    missing_core = []
    for m in CORE:
        ok = importlib.util.find_spec(m) is not None
        print(f"  {m:<22} {'+' if ok else '-'}")
        if not ok:
            missing_core.append(m)
    print()
    print("OPTIONAL")
    for m in OPTIONAL:
        ok = importlib.util.find_spec(m) is not None
        print(f"  {m:<22} {'+' if ok else '-'}")
    print()
    if missing_core:
        print(f"MISSING CORE: {missing_core}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
