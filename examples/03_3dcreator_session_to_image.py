"""Render an IronEngine-3DCreator session JSON to a PNG.

Loads the session via 3DCreator's own ``Session.load``, replays the stored
``GenerationSpec`` through 3DCreator's deterministic procedural generator
(no LLM — the spec was resolved when the session was saved), and renders
the resulting ``GenerationResult`` through BonaFide.

Note: 3DCreator's ``Session`` dataclass (core/session.py:29-35) stores
``requirements`` / ``spec`` / ``seed`` / ``edit_history`` — there is no
``last_result`` field, so the cloud is regenerated from ``session.spec``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ironengine_bonafide.api import (
    DirectionalLight, Engine, PerspectiveCamera, PointCloud, RenderConfig,
    Scene, render,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True,
                        help=".iecreator.json session file")
    parser.add_argument("--out", type=Path, default=Path("preview.png"))
    args = parser.parse_args()

    try:
        from ironengine_3d_creator.core.session import Session                  # type: ignore[import-not-found]
        from ironengine_3d_creator.alignment.schema import GenerationSpec       # type: ignore[import-not-found]
        from ironengine_3d_creator.generation.compositor import generate        # type: ignore[import-not-found]
    except ImportError:
        print("ERROR: ironengine_3d_creator not installed.", file=sys.stderr)
        sys.exit(2)

    session = Session.load(args.session)
    if not session.spec:
        print("ERROR: session has no resolved spec; run 3DCreator's pipeline "
              "first so the session captures a GenerationSpec.", file=sys.stderr)
        sys.exit(3)

    # Deterministic replay of the saved spec (seed lives in the spec; the
    # session-level seed is the fallback for older sessions).
    spec = GenerationSpec.from_json(session.spec)
    if not spec.seed and session.seed:
        spec.seed = int(session.seed)
    result = generate(spec)                                                     # GenerationResult

    cloud = PointCloud.from_generation_result(result)
    cloud = cloud.with_lod().with_surfels().with_completion()

    engine = Engine.auto()
    scene = (Scene().add(cloud)
             .add(DirectionalLight(direction=(-0.4, -1.0, -0.3), intensity=3.0)))
    cam = PerspectiveCamera(position=(2.5, 1.8, 2.5), look_at=(0, 0.5, 0))
    cfg = RenderConfig(width=1280, height=720, output_color_space="sRGB")
    out = render(engine, scene, cam, cfg)

    # Tonemap contract: with output_color_space="sRGB", out.rgb is already
    # final display-ready sRGB — convert directly, never re-apply ACES.
    import imageio.v3 as iio
    arr = out.rgb.detach().clamp(0.0, 1.0).cpu().numpy()
    iio.imwrite(str(args.out), np.rint(arr * 255.0).astype(np.uint8))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
