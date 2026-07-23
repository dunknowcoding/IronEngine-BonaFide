"""Replace IronEngine-Sim's RenderWorld with the BonaFide path."""
from __future__ import annotations

import argparse
from pathlib import Path

from ironengine_bonafide.integrations.sim import install


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    args = parser.parse_args()

    install()                                # patch ironengine_sim.RenderWorld

    from ironengine_sim import SimSession    # noqa: E402
    session = SimSession.from_template(args.scene.stem, robots=[])
    session.world.render.render_viewport()
    print(f"BonaFide-rendered Sim viewport, last frame:",
          session.world.render._last_bonafide_frame.shape)
    session.close()


if __name__ == "__main__":
    main()
