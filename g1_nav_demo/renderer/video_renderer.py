from __future__ import annotations

import logging
import math
from typing import Optional

import mujoco
import numpy as np

logger = logging.getLogger(__name__)


class VideoRenderer:
    def __init__(
        self,
        model: mujoco.MjModel,
        output_path: str = "demo_output.mp4",
        fps: int = 30,
        width: int = 1280,
        height: int = 480,
        birdseye_camera: str = "birdseye",
        head_camera: str = "head",
        crf: int = 22,
        head_fraction: float = 0.6,
    ) -> None:
        self.model = model
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.birdseye_camera = birdseye_camera
        self.head_camera = head_camera
        self.crf = crf

        self._head_width = int(width * head_fraction)
        self._birdseye_width = width - self._head_width
        self._panel_height = height

        self._birdseye_cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, birdseye_camera
        )
        self._head_cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, head_camera
        )

        self._birdseye_renderer = mujoco.Renderer(
            model, height=self._panel_height, width=self._birdseye_width
        )
        self._head_renderer = mujoco.Renderer(
            model, height=self._panel_height, width=self._head_width
        )

        self._container = None
        self._stream = None
        self._closed = False

        self.hazard_banner: str | None = None
        self.safe_banner: str | None = None
        self.obstacle_banner: str | None = None

    def _init_video_writer(self, first_frame: np.ndarray) -> None:
        import av

        self._container = av.open(self.output_path, mode="w")
        self._stream = self._container.add_stream("h264", rate=self.fps)
        self._stream.width = self.width
        self._stream.height = self.height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"crf": str(self.crf)}

    _CHASE_DIST: float = 2.5    # metres behind robot
    _CHASE_HEIGHT: float = 1.5  # metres above robot base

    @staticmethod
    def _rot_to_quat(R: np.ndarray) -> np.ndarray:
        """Convert 3×3 rotation matrix (columns = camera axes in world) → quaternion (w,x,y,z)."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            return np.array([0.25 / s,
                             (R[2, 1] - R[1, 2]) * s,
                             (R[0, 2] - R[2, 0]) * s,
                             (R[1, 0] - R[0, 1]) * s])
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                             (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
        if R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                             0.25 * s, (R[1, 2] + R[2, 1]) / s])
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                         (R[1, 2] + R[2, 1]) / s, 0.25 * s])

    def _update_head_camera(self, data: mujoco.MjData, body_id: int) -> None:
        """3rd-person chase camera: follows robot from behind, looks at torso."""
        pos = data.xpos[body_id].copy()
        quat = data.xquat[body_id].copy()

        # Extract yaw only so the camera doesn't wobble with locomotion roll/pitch.
        w, x, y, z = quat
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        cam_pos = np.array([
            pos[0] - self._CHASE_DIST * math.cos(yaw),
            pos[1] - self._CHASE_DIST * math.sin(yaw),
            pos[2] + self._CHASE_HEIGHT,
        ])

        # Look-at: aim at robot's torso (slightly above base)
        target = pos + np.array([0.0, 0.0, 0.5])
        look = target - cam_pos
        look /= np.linalg.norm(look)

        # Build orthonormal camera frame: cam looks along local -Z.
        cam_z = -look                                         # cam Z in world
        world_up = np.array([0.0, 0.0, 1.0])
        cam_y = world_up - np.dot(world_up, cam_z) * cam_z   # Gram-Schmidt
        cam_y /= np.linalg.norm(cam_y)
        cam_x = np.cross(cam_y, cam_z)                       # right axis

        R = np.column_stack([cam_x, cam_y, cam_z])
        cam_quat = self._rot_to_quat(R)

        self.model.cam_pos[self._head_cam_id] = cam_pos
        self.model.cam_quat[self._head_cam_id] = cam_quat

    def _update_birdseye_camera(self, data: mujoco.MjData, body_id: int) -> None:
        pass

    def _overlay_text(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple[int, int],
        font_scale: float = 0.7,
        color: tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        import cv2

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(
            bgr,
            (position[0] - 2, position[1] - text_h - 4),
            (position[0] + text_w + 2, position[1] + baseline + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(bgr, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def _overlay_label(
        self,
        panel: np.ndarray,
        label: str,
    ) -> np.ndarray:
        import cv2

        bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thick = 1
        (tw, th), bl = cv2.getTextSize(label, font, scale, thick)
        cv2.rectangle(bgr, (4, 4), (tw + 12, th + bl + 8), (0, 0, 0), -1)
        cv2.putText(bgr, label, (8, th + 6), font, scale, (100, 255, 100), thick, cv2.LINE_AA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _overlay_banner(
        self, frame: np.ndarray, text: str, bgr_color: tuple[int, int, int]
    ) -> np.ndarray:
        import cv2

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w, _ = bgr.shape
        cv2.rectangle(bgr, (0, 0), (w, 60), bgr_color, -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.9
        thick = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        x = max(10, (w - tw) // 2)
        y = 40
        cv2.putText(bgr, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _world_to_birdseye_pixel(self, wx: float, wy: float) -> tuple[int, int]:
        """Project a world XY position onto the birdseye panel in pixels."""
        cam_pos = self.model.cam_pos[self._birdseye_cam_id]
        cx, cy, ch = float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])
        fovy_rad = math.radians(float(self.model.cam_fovy[self._birdseye_cam_id]))
        half_h = math.tan(fovy_rad / 2.0) * ch
        half_w = half_h * (self._birdseye_width / self._panel_height)
        px = (wx - (cx - half_w)) / (2.0 * half_w) * self._birdseye_width
        py = ((cy + half_h) - wy) / (2.0 * half_h) * self._panel_height
        return int(px), int(py)

    def _overlay_waypoints(
        self,
        panel: np.ndarray,
        waypoints: list[tuple[float, float]],
        current_wp_idx: int,
    ) -> np.ndarray:
        import cv2

        bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        for i, (wx, wy) in enumerate(waypoints):
            px, py = self._world_to_birdseye_pixel(wx, wy)
            is_final = (i == len(waypoints) - 1)
            is_current = (i == current_wp_idx)
            if is_final:
                # Big red goal dot
                cv2.circle(bgr, (px, py), 10, (0, 0, 220), -1)
                cv2.circle(bgr, (px, py), 10, (255, 255, 255), 2)
            elif is_current:
                # Medium orange — next waypoint to reach
                cv2.circle(bgr, (px, py), 6, (0, 140, 255), -1)
                cv2.circle(bgr, (px, py), 6, (255, 255, 255), 1)
            else:
                # Small grey — completed or future intermediate
                cv2.circle(bgr, (px, py), 4, (120, 120, 120), -1)
        # Draw path line connecting waypoints
        if len(waypoints) > 1:
            pts = [self._world_to_birdseye_pixel(wx, wy) for wx, wy in waypoints]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(bgr, a, b, (80, 80, 80), 1)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def render_frame(
        self,
        data: mujoco.MjData,
        command: str = "",
        distance: Optional[float] = None,
        update_head_camera: bool = True,
        head_body_id: Optional[int] = None,
        goal_waypoints: Optional[list[tuple[float, float]]] = None,
        current_wp_idx: int = 0,
    ) -> np.ndarray:
        if update_head_camera and head_body_id is not None:
            self._update_head_camera(data, head_body_id)
            self._update_birdseye_camera(data, head_body_id)

        self._birdseye_renderer.update_scene(data, camera=self._birdseye_cam_id)
        self._head_renderer.update_scene(data, camera=self._head_cam_id)

        birdseye_frame = self._birdseye_renderer.render()
        head_frame = self._head_renderer.render()

        if goal_waypoints:
            birdseye_frame = self._overlay_waypoints(birdseye_frame, goal_waypoints, current_wp_idx)

        birdseye_frame = self._overlay_label(birdseye_frame, "OVERHEAD")
        head_frame = self._overlay_label(head_frame, "3RD PERSON")

        combined = np.hstack([birdseye_frame, head_frame])

        if command:
            combined = self._overlay_text(
                combined, f"Command: {command}", (10, 30), font_scale=0.7, color=(255, 255, 255)
            )

        if distance is not None:
            combined = self._overlay_text(
                combined,
                f"Distance: {distance:.2f}m",
                (10, 60),
                font_scale=0.7,
                color=(255, 255, 255),
            )

        if self.hazard_banner:
            combined = self._overlay_banner(combined, self.hazard_banner, (0, 0, 220))
        elif self.safe_banner:
            combined = self._overlay_banner(combined, self.safe_banner, (34, 139, 34))
        elif self.obstacle_banner:
            combined = self._overlay_banner(combined, self.obstacle_banner, (0, 200, 255))

        return combined

    def write_frame(self, frame: np.ndarray) -> None:
        import av

        if self._closed:
            raise RuntimeError("VideoRenderer is closed")

        if self._container is None:
            self._init_video_writer(frame)

        frame_av = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in self._stream.encode(frame_av):
            self._container.mux(packet)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._container is not None and self._stream is not None:
            for packet in self._stream.encode():
                self._container.mux(packet)
            self._container.close()

        self._birdseye_renderer.close()
        self._head_renderer.close()

    def snapshot(
        self,
        camera_name: str,
        data: mujoco.MjData,
        width: int = 640,
        height: int = 480,
    ) -> bytes:
        """Render a single MuJoCo camera and return PNG-encoded bytes.

        Creates a fresh Renderer at the requested size and closes it
        immediately, so this does not interfere with the active panel
        renderers used by render_frame().
        """
        if self._closed:
            raise RuntimeError("VideoRenderer is closed")

        import io

        from PIL import Image

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"Unknown camera: {camera_name!r}")

        renderer = mujoco.Renderer(self.model, height=height, width=width)
        try:
            renderer.update_scene(data, camera=cam_id)
            rgb = renderer.render()
        finally:
            renderer.close()

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return buf.getvalue()
