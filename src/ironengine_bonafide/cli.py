"""`bonafide` command-line interface.

Subcommands:

  bonafide render <scene.json> --out <img.png> [--config <cfg.json>]
  bonafide bundle <bundle.bnf>  --out <img.png>
  bonafide info                                  # backend probe
  bonafide list-templates                        # bundled examples

The "scene file" is a tiny JSON describing point cloud / mesh paths and
camera. Use it for one-shots; for richer scenes use the Python API.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import torch

from ironengine_bonafide.api import (
    DirectionalLight,
    Engine,
    Mesh,
    OrthographicCamera,
    PerspectiveCamera,
    PointCloud,
    RenderConfig,
    Scene,
    render,
)
from ironengine_bonafide.bundle import RenderBundle
from ironengine_bonafide.core.backend import probe
from ironengine_bonafide.logging import logger


def _build_scene_from_json(spec: dict[str, Any]) -> tuple[Scene, PerspectiveCamera | OrthographicCamera]:
    scene = Scene(name=spec.get("name", "scene"))
    for entry in spec.get("pointclouds", []):
        path = Path(entry["path"])
        cloud = PointCloud.from_ply(path) if path.suffix.lower() == ".ply" else PointCloud.from_pcd(path)
        if entry.get("lod"):
            cloud = cloud.with_lod()
        if entry.get("completion"):
            cloud = cloud.with_completion()
        if entry.get("gsplat"):
            cloud = cloud.with_gsplat()
        if entry.get("surfels"):
            cloud = cloud.with_surfels()
        scene.add(cloud)
    for entry in spec.get("meshes", []):
        path = Path(entry["path"])
        if path.suffix.lower() == ".obj":
            mesh = Mesh.from_obj(path)
        elif path.suffix.lower() in (".glb", ".gltf"):
            mesh = Mesh.from_glb(path)
        elif path.suffix.lower() == ".ply":
            from ironengine_bonafide.assets.loaders.ply import load_mesh
            mesh = load_mesh(path)
        else:
            raise ValueError(f"Unsupported mesh format: {path.suffix}")
        scene.add(mesh)
    for entry in spec.get("lights", []):
        if entry.get("kind", "directional") == "directional":
            scene.add(DirectionalLight(
                direction=tuple(entry.get("direction", (-0.4, -1.0, -0.3))),    # type: ignore[arg-type]
                color=tuple(entry.get("color", (1.0, 0.98, 0.95))),             # type: ignore[arg-type]
                intensity=float(entry.get("intensity", 3.0)),
            ))
    cam_spec = spec.get("camera", {})
    cam = PerspectiveCamera(
        position=tuple(cam_spec.get("position", (3, 2, 3))),                    # type: ignore[arg-type]
        look_at=tuple(cam_spec.get("look_at", (0, 0, 0))),                      # type: ignore[arg-type]
        fov_deg=float(cam_spec.get("fov_deg", 45.0)),
    )
    return scene, cam


def cmd_render(args: argparse.Namespace) -> int:
    scene_spec = json.loads(Path(args.scene).read_text(encoding="utf-8"))
    scene, cam = _build_scene_from_json(scene_spec)
    cfg = RenderConfig.from_file(args.config) if args.config else RenderConfig()
    if args.width:
        cfg.width = int(args.width)
    if args.height:
        cfg.height = int(args.height)
    engine = _engine_from_choice(args.backend)
    out = render(engine, scene, cam, cfg)
    img = _display_uint8(out, cfg.exposure)
    iio.imwrite(args.out, img)
    logger.info(f"wrote {args.out}  ({img.shape[1]}x{img.shape[0]})")
    if out.skipped_passes:
        logger.info(f"skipped passes: {out.skipped_passes}")
    return 0


def _display_uint8(out: Any, exposure: float = 1.0) -> Any:
    """uint8 image for saving.

    With ``output_color_space="sRGB"`` the render tensor is already final
    display-ready sRGB (ACES + encoding applied in TonemapPass) — encode
    it directly. Applying ACES again would double-tonemap. Linear-HDR
    output still goes through the standard ACES display path here.
    """
    if out.color_space == "sRGB":
        return out.rgb.to_uint8_display().detach().cpu().numpy()
    return out.rgb.to_aces_srgb_uint8(exposure=exposure).detach().cpu().numpy()


def cmd_bundle(args: argparse.Namespace) -> int:
    bundle = RenderBundle.load(args.bundle)
    engine = _engine_from_choice(args.backend)
    out = bundle.reproduce(engine)
    img = _display_uint8(out)
    iio.imwrite(args.out, img)
    logger.info(f"reproduced bundle → {args.out}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    p = probe()
    print(f"cuda_available  = {p.cuda_available}")
    print(f"wgpu_available  = {p.wgpu_available}")
    print(f"torch_cuda      = {p.torch_cuda}")
    print(f"torch_mps       = {p.torch_mps}")
    print(f"torch_version   = {torch.__version__}")
    if p.notes:
        print("notes:")
        for n in p.notes:
            print(f"  - {n}")
    return 0


def cmd_list_templates(_args: argparse.Namespace) -> int:
    here = Path(__file__).parent.parent.parent / "examples" / "assets"
    if not here.exists():
        print(f"(no example assets at {here})")
        return 0
    for p in sorted(here.iterdir()):
        if p.is_file():
            print(p.name)
    return 0


def _engine_from_choice(choice: str) -> Engine:
    if choice == "auto":
        return Engine.auto()
    if choice == "cuda":
        return Engine.cuda()
    if choice == "wgpu":
        return Engine.wgpu()
    if choice == "cpu":
        return Engine.cpu()
    raise ValueError(f"Unknown backend choice: {choice}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bonafide", description="IronEngine-BonaFide CLI")
    parser.add_argument("--backend", choices=("auto", "cuda", "wgpu", "cpu"), default="auto")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_render = sub.add_parser("render", help="render a JSON scene to PNG")
    p_render.add_argument("scene")
    p_render.add_argument("--out", required=True)
    p_render.add_argument("--config", default=None)
    p_render.add_argument("--width", type=int, default=None)
    p_render.add_argument("--height", type=int, default=None)
    p_render.set_defaults(func=cmd_render)

    p_bundle = sub.add_parser("bundle", help="reproduce a saved render bundle (.bnf)")
    p_bundle.add_argument("bundle")
    p_bundle.add_argument("--out", required=True)
    p_bundle.set_defaults(func=cmd_bundle)

    p_info = sub.add_parser("info", help="probe available backends")
    p_info.set_defaults(func=cmd_info)

    p_list = sub.add_parser("list-templates", help="list bundled example assets")
    p_list.set_defaults(func=cmd_list_templates)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
