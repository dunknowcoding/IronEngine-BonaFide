"""RenderConfig must round-trip through JSON."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ironengine_bonafide.api import RenderConfig
from ironengine_bonafide.core.config import (
    CompletionConfig, FogConfig, GsplatConfig, LodConfig, SurfelConfig,
)


def test_dict_round_trip() -> None:
    cfg = RenderConfig(
        width=1920, height=1080, samples=4, aa="taa",
        gsplat=GsplatConfig(enabled=True, sigma_scale=1.5, densify=False),
        lod=LodConfig(enabled=True, screen_space_error_px=2.0),
        completion=CompletionConfig(enabled=True, mlp_width=128, mlp_depth=4),
        surfels=SurfelConfig(enabled=True, radius_factor=2.0),
        fog=FogConfig(enabled=True, density=0.05),
        sensor_outputs=("rgb", "depth", "normals"),
        seed=1234,
    )
    d = cfg.to_dict()
    cfg2 = RenderConfig.from_dict(d)
    assert cfg2.width == cfg.width
    assert cfg2.gsplat.sigma_scale == 1.5
    assert cfg2.completion.mlp_width == 128
    assert cfg2.surfels.radius_factor == 2.0
    assert cfg2.fog.density == 0.05
    assert cfg2.seed == 1234


def test_json_file_round_trip(tmp_path: Path) -> None:
    cfg = RenderConfig(width=800, height=600, neural_denoise=True)
    p = tmp_path / "cfg.json"
    cfg.to_file(p)
    cfg2 = RenderConfig.from_file(p)
    assert cfg2.width == 800
    assert cfg2.neural_denoise is True


def test_json_serialisable() -> None:
    cfg = RenderConfig()
    json.dumps(cfg.to_dict())                   # must not raise
