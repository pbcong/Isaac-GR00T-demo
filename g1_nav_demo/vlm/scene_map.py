from __future__ import annotations

import io
import math

from PIL import Image, ImageDraw

_CAM_X, _CAM_Y, _CAM_H = 1.5, 0.0, 12.0
_CAM_FOVY_DEG = 50.0

_GRID_SPACING = 0.5
_LABEL_SPACING = 1.0


def _cam_w2p(wx: float, wy: float, width: int, height: int) -> tuple[int, int]:
    half_h = math.tan(math.radians(_CAM_FOVY_DEG / 2)) * _CAM_H
    half_w = half_h * (width / height)
    px = (wx - (_CAM_X - half_w)) / (2 * half_w) * width
    py = ((_CAM_Y + half_h) - wy) / (2 * half_h) * height
    return int(px), int(py)


def render_mujoco_frame(model, data, camera: str = "birdseye",
                        width: int = 640, height: int = 640) -> bytes:
    import mujoco
    import numpy as np

    renderer = mujoco.Renderer(model, height=height, width=width)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render()
    renderer.close()

    img = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(img)

    half_h = math.tan(math.radians(_CAM_FOVY_DEG / 2)) * _CAM_H
    half_w = half_h * (width / height)
    x0 = math.floor(_CAM_X - half_w)
    x1 = math.ceil(_CAM_X + half_w)
    y0 = math.floor(_CAM_Y - half_h)
    y1 = math.ceil(_CAM_Y + half_h)

    def w2p(wx, wy):
        return _cam_w2p(wx, wy, width, height)

    gx = x0
    while gx <= x1:
        is_label = abs(gx - round(gx / _LABEL_SPACING) * _LABEL_SPACING) < 1e-9
        is_major = abs(gx) < 1e-9
        p1 = w2p(gx, y0)
        p2 = w2p(gx, y1)
        lw = 2 if is_major else 1
        col = (200, 200, 200) if is_major else (130, 130, 130) if is_label else (90, 90, 90)
        draw.line([p1, p2], fill=col, width=lw)
        if is_label:
            lp = w2p(gx, _CAM_Y - half_h + 0.4)
            draw.text((lp[0] - 5, lp[1]), str(gx), fill=(255, 255, 100))
        gx += _GRID_SPACING

    gy = y0
    while gy <= y1:
        is_label = abs(gy - round(gy / _LABEL_SPACING) * _LABEL_SPACING) < 1e-9
        is_major = abs(gy) < 1e-9
        p1 = w2p(x0, gy)
        p2 = w2p(x1, gy)
        lw = 2 if is_major else 1
        col = (200, 200, 200) if is_major else (130, 130, 130) if is_label else (90, 90, 90)
        draw.line([p1, p2], fill=col, width=lw)
        if is_label:
            lp = w2p(_CAM_X - half_w + 0.2, gy)
            draw.text((lp[0], lp[1] - 7), str(gy), fill=(255, 255, 100))
        gy += _GRID_SPACING

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
