"""PLY loader: CRLF-terminated headers (Windows-authored files) parse the
same as LF-terminated ones, for both the point-cloud and mesh paths."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ironengine_bonafide.assets.loaders.ply import load_mesh, load_pointcloud

_CLOUD = (
    "ply\r\nformat ascii 1.0\r\n"
    "element vertex 2\r\n"
    "property float x\r\nproperty float y\r\nproperty float z\r\n"
    "property uchar red\r\nproperty uchar green\r\nproperty uchar blue\r\n"
    "end_header\r\n"
    "0 0 0 255 0 0\r\n"
    "1 1 1 0 255 0\r\n"
)

_MESH = (
    "ply\r\nformat ascii 1.0\r\n"
    "element vertex 3\r\n"
    "property float x\r\nproperty float y\r\nproperty float z\r\n"
    "element face 1\r\n"
    "property list uchar int vertex_indices\r\n"
    "end_header\r\n"
    "0 0 0\r\n1 0 0\r\n0 1 0\r\n"
    "3 0 1 2\r\n"
)


def test_crlf_pointcloud(tmp_path: Path) -> None:
    p = tmp_path / "cloud.ply"
    p.write_bytes(_CLOUD.encode())
    cloud = load_pointcloud(p)
    assert cloud.num_points == 2
    np.testing.assert_allclose(cloud.positions.numpy()[1], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(cloud.colors.numpy()[0], [1.0, 0.0, 0.0])


def test_crlf_mesh(tmp_path: Path) -> None:
    p = tmp_path / "tri.ply"
    p.write_bytes(_MESH.encode())
    mesh = load_mesh(p)
    assert mesh.num_vertices == 3
    assert mesh.num_triangles == 1
    np.testing.assert_array_equal(mesh.indices.numpy(), np.array([[0, 1, 2]]))
