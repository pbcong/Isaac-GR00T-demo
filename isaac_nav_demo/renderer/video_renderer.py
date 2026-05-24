"""Video renderer for the Isaac Sim warehouse demo.

Produces the same side-by-side output as the MuJoCo version:
  [birdseye overhead | 3rd-person chase cam]

Cameras are Isaac omni.isaac.sensor.Camera objects.  The chase camera pose
is updated each frame to follow the robot.  A separate onboard camera
(parented to the robot's head link) is used for look() snapshots.

Banner and overlay logic is identical to the MuJoCo renderer.
"""
from __future__ import annotations

import io
import logging
import math
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from omni.isaac.core import World

logger = logging.getLogger(__name__)


class VideoRenderer:
    _CHASE_DIST: float = 3.0    # metres behind robot
    _CHASE_HEIGHT: float = 1.8  # metres above robot base

    def __init__(
        self,
        world: "World",
        output_path: str = "demo_output.mp4",
        fps: int = 30,
        width: int = 1280,
        height: int = 480,
        head_fraction: float = 0.6,
        crf: int = 22,
    ) -> None:
        self.world = world
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.crf = crf

        self._head_width    = int(width * head_fraction)
        self._birdseye_width = width - self._head_width
        self._panel_height  = height

        self.hazard_banner:   str | None = None
        self.safe_banner:     str | None = None
        self.obstacle_banner: str | None = None

        self._container = None
        self._stream    = None
        self._closed    = False

        self._birdseye_cam = None
        self._chase_cam    = None
        self._onboard_cam  = None
        self._cameras_initialized = False

    # ------------------------------------------------------------------
    # Camera setup (deferred until first frame so world is fully loaded)
    # ------------------------------------------------------------------

    def _init_cameras(self) -> None:
        from omni.isaac.sensor import Camera
        import omni.isaac.core.utils.rotations as rot_utils

        # Overhead birdseye camera — fixed position above warehouse centre
        self._birdseye_cam = Camera(
            prim_path="/World/Cameras/Birdseye",
            position=np.array([10.0, 0.0, 22.0]),
            # Look straight down: rotate -90° around X
            orientation=rot_utils.euler_angles_to_quats(
                np.array([-90.0, 0.0, 0.0]), degrees=True
            ),
            resolution=(self._birdseye_width, self._panel_height),
            frequency=self.fps,
        )

        # Chase camera — pose updated each frame
        self._chase_cam = Camera(
            prim_path="/World/Cameras/Chase",
            resolution=(self._head_width, self._panel_height),
            frequency=self.fps,
        )

        # Onboard head camera for look() snapshots.
        # Parented under the robot's head link so it moves with the robot.
        self._onboard_cam = Camera(
            prim_path="/World/G1/head_link/OnboardCam",
            position=np.array([0.04, 0.0, 0.0]),   # 4 cm forward of head origin
            orientation=rot_utils.euler_angles_to_quats(
                np.array([0.0, -15.0, 0.0]), degrees=True  # slight downward tilt
            ),
            resolution=(1280, 960),
            frequency=self.fps,
        )

        for cam in [self._birdseye_cam, self._chase_cam, self._onboard_cam]:
            cam.initialize()

        self._cameras_initialized = True
        logger.info("Isaac cameras initialised")

    # ------------------------------------------------------------------
    # Chase camera pose
    # ------------------------------------------------------------------

    def _update_chase_camera(self, robot_pos: np.ndarray, robot_yaw: float) -> None:
        import omni.isaac.core.utils.rotations as rot_utils

        cam_pos = np.array([
            robot_pos[0] - self._CHASE_DIST * math.cos(robot_yaw),
            robot_pos[1] - self._CHASE_DIST * math.sin(robot_yaw),
            robot_pos[2] + self._CHASE_HEIGHT,
        ])

        target = robot_pos + np.array([0.0, 0.0, 0.5])
        look   = target - cam_pos
        look  /= np.linalg.norm(look)

        # Build camera quaternion: camera -Z faces along look
        cam_z     = -look
        world_up  = np.array([0.0, 0.0, 1.0])
        cam_y     = world_up - np.dot(world_up, cam_z) * cam_z
        cam_y    /= np.linalg.norm(cam_y)
        cam_x     = np.cross(cam_y, cam_z)
        R         = np.column_stack([cam_x, cam_y, cam_z])
        quat      = _rot_to_quat(R)   # (w, x, y, z)

        self._chase_cam.set_world_pose(
            position=cam_pos,
            orientation=quat,
        )

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------

    def render_frame(
        self,
        robot_pos: np.ndarray,
        robot_yaw: float,
        command: str = "",
        distance: Optional[float] = None,
        goal_waypoints: Optional[list[tuple[float, float]]] = None,
        current_wp_idx: int = 0,
    ) -> np.ndarray:
        if not self._cameras_initialized:
            self._init_cameras()

        self._update_chase_camera(robot_pos, robot_yaw)

        # Tick render so camera buffers refresh
        self.world.render()

        birdseye_rgba = self._birdseye_cam.get_rgba()
        chase_rgba    = self._chase_cam.get_rgba()

        birdseye_frame = birdseye_rgba[:, :, :3].astype(np.uint8)
        chase_frame    = chase_rgba[:, :, :3].astype(np.uint8)

        if goal_waypoints:
            birdseye_frame = self._overlay_waypoints(
                birdseye_frame, goal_waypoints, current_wp_idx
            )

        birdseye_frame = self._overlay_label(birdseye_frame, "OVERHEAD")
        chase_frame    = self._overlay_label(chase_frame,    "3RD PERSON")

        combined = np.hstack([birdseye_frame, chase_frame])

        if command:
            combined = self._overlay_text(
                combined, f"Command: {command}", (10, 30), font_scale=0.7
            )
        if distance is not None:
            combined = self._overlay_text(
                combined, f"Distance: {distance:.2f}m", (10, 60), font_scale=0.7
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
            self._init_video_writer()

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
        logger.info("VideoRenderer closed: %s", self.output_path)

    def snapshot(self, width: int = 1280, height: int = 960) -> bytes:
        """Capture head-camera PNG for the look() tool."""
        if self._closed:
            raise RuntimeError("VideoRenderer is closed")
        if not self._cameras_initialized:
            self._init_cameras()

        self.world.render()
        rgba = self._onboard_cam.get_rgba()
        rgb  = rgba[:, :, :3].astype(np.uint8)

        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Video writer init
    # ------------------------------------------------------------------

    def _init_video_writer(self) -> None:
        import av

        self._container = av.open(self.output_path, mode="w")
        self._stream    = self._container.add_stream("h264", rate=self.fps)
        self._stream.width   = self.width
        self._stream.height  = self.height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"crf": str(self.crf)}

    # ------------------------------------------------------------------
    # Overlay helpers (identical logic to MuJoCo renderer)
    # ------------------------------------------------------------------

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
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(
            bgr,
            (position[0] - 2, position[1] - th - 4),
            (position[0] + tw + 2, position[1] + baseline + 2),
            (0, 0, 0), -1,
        )
        cv2.putText(bgr, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _overlay_label(self, panel: np.ndarray, label: str) -> np.ndarray:
        import cv2

        bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.5, 1
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
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.putText(bgr, text, (max(10, (w - tw) // 2), 40), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _world_to_birdseye_pixel(
        self, wx: float, wy: float
    ) -> tuple[int, int]:
        # Birdseye camera is at (10, 0, 22), looking straight down, fovy ≈ 60°
        cam_h = 22.0
        fovy_rad = math.radians(60.0)
        half_h = math.tan(fovy_rad / 2.0) * cam_h
        half_w = half_h * (self._birdseye_width / self._panel_height)
        cam_cx, cam_cy = 10.0, 0.0
        px = (wx - (cam_cx - half_w)) / (2.0 * half_w) * self._birdseye_width
        py = ((cam_cy + half_h) - wy) / (2.0 * half_h) * self._panel_height
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
            if i == len(waypoints) - 1:
                cv2.circle(bgr, (px, py), 10, (0, 0, 220), -1)
                cv2.circle(bgr, (px, py), 10, (255, 255, 255), 2)
            elif i == current_wp_idx:
                cv2.circle(bgr, (px, py), 6, (0, 140, 255), -1)
                cv2.circle(bgr, (px, py), 6, (255, 255, 255), 1)
            else:
                cv2.circle(bgr, (px, py), 4, (120, 120, 120), -1)
        if len(waypoints) > 1:
            pts = [self._world_to_birdseye_pixel(wx, wy) for wx, wy in waypoints]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(bgr, a, b, (80, 80, 80), 1)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion (w, x, y, z)."""
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
