"""Load the bundled SVGs as centered Matplotlib marker paths."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from fontTools.pens.recordingPen import RecordingPen
from fontTools.svgLib.path import parse_path
from matplotlib.path import Path as MplPath

from plotting.plot_style import fontawesome_icon_path, sharp_geometric_icon_path


def centered_unit_path(vertices, codes, *, flip_y=False) -> MplPath:
    """Center the vertex bounds, including curve controls but excluding CLOSEPOLY, and scale the longest side to 1."""
    vertices = np.asarray(vertices, dtype=float)
    codes = np.asarray(codes, dtype=np.uint8)
    drawable = vertices[codes != MplPath.CLOSEPOLY]
    minimum = drawable.min(axis=0)
    maximum = drawable.max(axis=0)
    center = 0.5 * (minimum + maximum)
    scale = float(np.max(maximum - minimum))
    normalized = (vertices - center) / scale
    if flip_y:
        normalized[:, 1] = -normalized[:, 1]
    return MplPath(normalized, codes)


def _svg_path_data(svg_path: str | Path) -> str:
    root = ET.parse(svg_path).getroot()
    path_element = root.find("{http://www.w3.org/2000/svg}path")
    if path_element is None:
        path_element = root.find("path")
    if path_element is None or "d" not in path_element.attrib:
        raise ValueError(f"No SVG path data found in {svg_path}")
    return path_element.attrib["d"]


def svg_marker_path(svg_path: str | Path) -> MplPath:
    """Convert the first SVG path to a normalized Matplotlib marker, reversing the SVG y direction."""
    pen = RecordingPen()
    parse_path(_svg_path_data(svg_path), pen)

    vertices = []
    codes = []
    subpath_start = (0.0, 0.0)
    for command, points in pen.value:
        if command == "moveTo":
            subpath_start = points[0]
            vertices.append(points[0])
            codes.append(MplPath.MOVETO)
        elif command == "lineTo":
            vertices.append(points[0])
            codes.append(MplPath.LINETO)
        elif command == "curveTo":
            vertices.extend(points)
            codes.extend([MplPath.CURVE4] * 3)
        elif command == "qCurveTo":
            vertices.extend(points)
            codes.extend([MplPath.CURVE3] * len(points))
        elif command == "closePath":
            vertices.append(subpath_start)
            codes.append(MplPath.CLOSEPOLY)
        else:
            raise ValueError(f"Unsupported SVG path command: {command}")
    return centered_unit_path(vertices, codes, flip_y=True)


@cache
def fontawesome_marker_path(name: str) -> MplPath:
    return svg_marker_path(fontawesome_icon_path(name))


@cache
def sharp_geometric_marker_path(name: str) -> MplPath:
    return svg_marker_path(sharp_geometric_icon_path(name))
