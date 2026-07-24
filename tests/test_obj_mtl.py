"""Regression: OBJ loader must honor mtllib/usemtl (.mtl parsing)."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from ironengine_bonafide.assets.loaders.obj import load_mesh, load_mtl

_OBJ = """\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vn 0 0 1
vt 0 0
vt 1 0
vt 1 1
vt 0 1
"""

_MTL = """\
newmtl red_paint
Kd 0.8 0.1 0.1
Ks 0.2 0.2 0.2
Ns 16.0
d 1.0
map_Kd textures/red.png

newmtl green_paint
Kd 0.1 0.7 0.2
Ns 2.0
"""


def _write(tmp_path: Path, obj_body: str, mtl: str | None = _MTL) -> Path:
    obj = tmp_path / "model.obj"
    obj.write_text(obj_body, encoding="utf-8")
    if mtl is not None:
        (tmp_path / "model.mtl").write_text(mtl, encoding="utf-8")
    return obj


def test_mtl_parser_fields(tmp_path: Path) -> None:
    mtl = tmp_path / "model.mtl"
    mtl.write_text(_MTL, encoding="utf-8")
    recs = load_mtl(mtl)
    assert set(recs) == {"red_paint", "green_paint"}
    red = recs["red_paint"]
    assert red.kd == pytest.approx((0.8, 0.1, 0.1))
    assert red.ks == pytest.approx((0.2, 0.2, 0.2))
    assert red.ns == pytest.approx(16.0)
    assert red.d == pytest.approx(1.0)
    assert red.map_kd is not None and red.map_kd.endswith("red.png")


def test_single_material_attached(tmp_path: Path) -> None:
    obj = _write(tmp_path, "mtllib model.mtl\nusemtl red_paint\n" + _OBJ + "f 1/1/1 2/2/1 3/3/1 4/4/1\n")
    mesh = load_mesh(obj)
    assert mesh.material.name == "red_paint"
    assert mesh.material.albedo == pytest.approx((0.8, 0.1, 0.1))
    # Ns=16 -> roughness = sqrt(2/18) = 1/3 (Blinn→GGX approximation).
    assert mesh.material.roughness == pytest.approx(math.sqrt(2.0 / 18.0))
    assert mesh.material.albedo_map is not None
    assert mesh.material.albedo_map.endswith("red.png")
    # Single material: no baked colors, the material record carries the tint.
    assert mesh.colors is None


def test_multi_material_bakes_vertex_colors(tmp_path: Path) -> None:
    body = (
        "mtllib model.mtl\n" + _OBJ
        + "usemtl red_paint\nf 1/1/1 2/2/1 3/3/1\n"
        + "usemtl green_paint\nf 1/1/1 3/3/1 4/4/1\n"
    )
    mesh = load_mesh(_write(tmp_path, body))
    # Majority tie broken deterministically; both tints appear in colors.
    assert mesh.colors is not None
    cols = {tuple(round(float(c), 3) for c in row) for row in mesh.colors}
    assert (0.8, 0.1, 0.1) in cols
    assert (0.1, 0.7, 0.2) in cols
    assert mesh.material.name in {"red_paint", "green_paint"}


def test_missing_mtllib_keeps_default(tmp_path: Path) -> None:
    obj = _write(tmp_path, _OBJ + "f 1/1/1 2/2/1 3/3/1 4/4/1\n", mtl=None)
    mesh = load_mesh(obj)
    assert mesh.material.name == "default"
    assert mesh.material.albedo == pytest.approx((0.8, 0.8, 0.8))
    assert mesh.colors is None


def test_missing_mtl_file_keeps_default(tmp_path: Path) -> None:
    # mtllib references a file that does not exist: warn and carry on.
    obj = _write(tmp_path, "mtllib nope.mtl\nusemtl red_paint\n" + _OBJ
                 + "f 1/1/1 2/2/1 3/3/1 4/4/1\n", mtl=None)
    mesh = load_mesh(obj)
    assert mesh.material.name == "default"
    assert mesh.colors is None
