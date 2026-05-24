"""Renders the overhead scene image used by VLMBridge for path planning.

Isaac Sim version: captures a frame from a high-altitude overhead camera,
then overlays a metric grid and coordinate labels — identical to the MuJoCo
version, just with a different image source.
"""
from __future__ import annotations

import io
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.core import World
    from omni.isaac.sensor import Camera

# Camera position used for the VLM scene image
_CAM_X, _CAM_Y, _CAM_H = 10.0, 0.0, 25.0
_CAM_FOVY_DEG = 60.0

_GRID_SPACING = 1.0
_LABEL_SPACING = 2.0


def _cam_w2p(wx: float, wy: float, width: int, height: int) -> tuple[int, int]:
    half_h = math.tan(math.radians(_CAM_FOVY_DEG / 2)) * _CAM_H
    half_w = half_h * (width / height)
    px = (wx - (_CAM_X - half_w)) / (2 * half_w) * width
    py = ((_CAM_Y + half_h) - wy) / (2 * half_h) * height
    return int(px), int(py)


def render_isaac_frame(
    world: "World",
    overhead_cam: "Camera",
    width: int = 640,
    height: int = 640,
) -> bytes:
    """Capture overhead camera + overlay grid, return PNG bytes."""
    from PIL import Image, ImageDraw

    world.render()
    rgba = overhead_cam.get_rgba()
    rgb  = rgba[:, :, :3].astype("uint8")

    img  = Image.fromarray(rgb).resize((width, height))
    draw = ImageDraw.Draw(img)

    half_h = math.tan(math.radians(_CAM_FOVY_DEG / 2)) * _CAM_H
    half_w = half_h * (width / height)
    x0 = math.floor(_CAM_X - half_w)
    x1 = math.ceil (_CAM_X + half_w)
    y0 = math.floor(_CAM_Y - half_h)
    y1 = math.ceil (_CAM_Y + half_h)

    def w2p(wx: float, wy: float) -> tuple[int, int]:
        return _cam_w2p(wx, wy, width, height)

    # Vertical grid lines
    gx = float(x0)
    while gx <= x1:
        is_label = abs(gx - round(gx / _LABEL_SPACING) * _LABEL_SPACING) < 1e-9
        is_major = abs(gx) < 1e-9
        col = (200, 200, 200) if is_major else (130, 130, 130) if is_label else (70, 70, 70)
        lw  = 2 if is_major else 1
        draw.line([w2p(gx, y0), w2p(gx, y1)], fill=col, width=lw)
        if is_label:
            lp = w2p(gx, _CAM_Y - half_h + 1.0)
            draw.text((lp[0] - 6, lp[1]), str(int(gx)), fill=(255, 255, 100))
        gx += _GRID_SPACING

    # Horizontal grid lines
    gy = float(y0)
    while gy <= y1:
        is_label = abs(gy - round(gy / _LABEL_SPACING) * _LABEL_SPACING) < 1e-9
        is_major = abs(gy) < 1e-9
        col = (200, 200, 200) if is_major else (130, 130, 130) if is_label else (70, 70, 70)
        lw  = 2 if is_major else 1
        draw.line([w2p(x0, gy), w2p(x1, gy)], fill=col, width=lw)
        if is_label:
            lp = w2p(_CAM_X - half_w + 0.5, gy)
            draw.text((lp[0], lp[1] - 8), str(int(gy)), fill=(255, 255, 100))
        gy += _GRID_SPACING

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
