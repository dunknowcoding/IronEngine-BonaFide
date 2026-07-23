"""Render a GLB / OBJ mesh to PNG."""
from __future__ import annotations

import argparse
from pathlib import Path

from ironengine_bonafide.api import (
    DirectionalLight, Engine, Mesh, PBRMaterial, PerspectiveCamera, RenderConfig,
    Scene, render,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--out", type=Path, default=Path("preview.png"))
    args = parser.parse_args()

    if args.mesh.suffix.lower() == ".obj":
        mesh = Mesh.from_obj(args.mesh)
    else:
        mesh = Mesh.from_glb(args.mesh)

    engine = Engine.auto()
    scene = (Scene()
             .add(mesh.with_material(PBRMaterial(albedo=(0.85, 0.55, 0.3),
                                                  roughness=0.45, metallic=0.0)))
             .add(DirectionalLight(direction=(-0.4, -1.0, -0.3), intensity=3.0)))
    cam = PerspectiveCamera(position=(2, 1.5, 2), look_at=(0, 0.5, 0), fov_deg=45)
    cfg = RenderConfig(width=1280, height=720, output_color_space="sRGB")
    out = render(engine, scene, cam, cfg)
    # output_color_space="sRGB" → tensor is already display-ready sRGB;
    # save it directly (no second ACES conversion).
    out.rgb.save(str(args.out), display_ready=out.color_space == "sRGB")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
