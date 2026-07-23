"""Render a PLY point cloud to PNG with the auto-selected backend."""
from __future__ import annotations

import argparse
from pathlib import Path

from ironengine_bonafide.api import (
    DirectionalLight, Engine, PerspectiveCamera, PointCloud, RenderConfig,
    Scene, render,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply", type=Path, help="input PLY")
    parser.add_argument("--out", type=Path, default=Path("preview.png"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    engine = Engine.auto()
    scene = (Scene()
             .add(PointCloud.from_ply(args.ply).with_lod().with_surfels())
             .add(DirectionalLight(direction=(-0.4, -1.0, -0.3), intensity=3.0)))
    cam = PerspectiveCamera(position=(2.5, 1.8, 2.5), look_at=(0, 0.5, 0), fov_deg=45)
    cfg = RenderConfig(width=args.width, height=args.height,
                       output_color_space="sRGB", samples=1)
    out = render(engine, scene, cam, cfg)
    # output_color_space="sRGB" → tensor is already display-ready sRGB;
    # save it directly (no second ACES conversion).
    out.rgb.save(str(args.out), display_ready=out.color_space == "sRGB")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
