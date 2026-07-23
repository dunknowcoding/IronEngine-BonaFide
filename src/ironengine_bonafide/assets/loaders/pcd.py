"""PCD (Point Cloud Library) loader — ASCII for now."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ironengine_bonafide.core.pointcloud import PointCloud


def load_pointcloud(path: Path) -> PointCloud:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    fields: list[str] = []
    n_points = 0
    is_ascii = True
    header_end = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("FIELDS"):
            fields = s.split()[1:]
        elif s.startswith("POINTS"):
            n_points = int(s.split()[-1])
        elif s.startswith("DATA"):
            is_ascii = "ascii" in s
            header_end = i + 1
            break
    if not is_ascii:
        raise ValueError(f"Binary PCD not supported yet: {path}")

    if n_points == 0:
        return PointCloud.from_arrays(np.zeros((0, 3), dtype=np.float32), name=path.stem)

    xi, yi, zi = fields.index("x"), fields.index("y"), fields.index("z")
    has_rgb = "rgb" in fields
    rgb_i = fields.index("rgb") if has_rgb else -1

    positions = np.empty((n_points, 3), dtype=np.float32)
    colors = np.empty((n_points, 3), dtype=np.float32) if has_rgb else None
    for i in range(n_points):
        tokens = lines[header_end + i].split()
        positions[i, 0] = float(tokens[xi])
        positions[i, 1] = float(tokens[yi])
        positions[i, 2] = float(tokens[zi])
        if colors is not None:
            rgb_f = float(tokens[rgb_i])
            rgb_int = int(rgb_f) if not np.isnan(rgb_f) else 0xFFFFFF
            colors[i, 0] = ((rgb_int >> 16) & 0xFF) / 255.0
            colors[i, 1] = ((rgb_int >> 8) & 0xFF) / 255.0
            colors[i, 2] = (rgb_int & 0xFF) / 255.0
    return PointCloud.from_arrays(positions, colors, name=path.stem)
