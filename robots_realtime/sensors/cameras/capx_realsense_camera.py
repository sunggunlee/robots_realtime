from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser.transforms as vtf
import yaml

from robots_realtime.sensors.cameras.camera import CameraData
from robots_realtime.sensors.cameras.realsense_camera import RealSenseCamera


@dataclass
class CapXRealSenseCamera(RealSenseCamera):
    """RealSense adapter that emits the camera contract expected by CaP-X.

    Output:
      images["left_rgb"]              RGB uint8 image
      depth_data                      float32 metric depth in meters
      intrinsic_data["left"]["intrinsics_matrix"]
      extrinsics                      T_base_camera metadata
    """

    extrinsics_file: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if not self.enable_depth:
            raise RuntimeError(
                "CapXRealSenseCamera requires enable_depth=True."
            )

        # CaP-X/SAM3 should receive a true RGB image rather than an
        # infrared fallback masquerading as RGB.
        if self._use_infrared:
            raise RuntimeError(
                "This RealSense opened without a true color stream. "
                "Do not use the infrared fallback for CaP-X."
            )

        rs = self._rs

        # Align depth pixels to the RGB/color pixel grid.
        self._align = rs.align(rs.stream.color)

        # RealSense depth frames contain device units. Convert to meters.
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())

        # CaP-X expects the calibration for the same image plane used by SAM3.
        color_profile = (
            self._profile
            .get_stream(rs.stream.color)
            .as_video_stream_profile()
        )
        intr = color_profile.get_intrinsics()

        K = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.intrinsic_data = {
            "left": {
                "intrinsics_matrix": K,
                "distortion_coefficients": list(intr.coeffs),
                "distortion_model": str(intr.model).replace(
                    "distortion.", ""
                ),
            }
        }

        self.extrinsics = self._load_capx_extrinsics()

        print(
            "[CapXRealSenseCamera] ready "
            f"depth_scale={self._depth_scale} m/unit"
        )

    def _load_capx_extrinsics(self) -> dict | None:
        if self.extrinsics_file is None:
            return None

        path = Path(self.extrinsics_file)

        if not path.exists():
            raise FileNotFoundError(
                f"RealSense extrinsics file not found: {path}"
            )

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        position = np.asarray(
            data["position"], dtype=np.float64
        )
        rpy = np.asarray(
            data["rpy_radians"], dtype=np.float64
        )

        if position.shape != (3,):
            raise ValueError(
                f"position must contain 3 values, got {position}"
            )

        if rpy.shape != (3,):
            raise ValueError(
                f"rpy_radians must contain 3 values, got {rpy}"
            )

        # Match the existing ZED extrinsics convention exactly.
        wxyz = vtf.SO3.from_rpy_radians(*rpy).wxyz
        pose_mat = vtf.SE3(
            wxyz_xyz=np.concatenate([wxyz, position])
        ).as_matrix()

        return {
            "position": position,
            "wxyz": wxyz,
            "pose_mat": pose_mat,
        }

    def read(self) -> CameraData:
        if self._pipeline is None:
            raise RuntimeError(
                "RealSense pipeline is not running."
            )

        frames = self._pipeline.wait_for_frames(
            timeout_ms=1000
        )

        # Crucial: make depth correspond pixel-for-pixel to RGB.
        aligned = self._align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame:
            raise RuntimeError(
                "RealSense returned no color frame."
            )

        if not depth_frame:
            raise RuntimeError(
                "RealSense returned no depth frame."
            )

        rgb = np.asanyarray(
            color_frame.get_data()
        )

        depth_raw = np.asanyarray(
            depth_frame.get_data()
        )

        depth_m = (
            depth_raw.astype(np.float32)
            * self._depth_scale
        )

        # RealSense uses 0 for invalid depth.
        # CaP-X treats NaN as invalid, so normalize it here.
        depth_m[depth_raw == 0] = np.nan

        if depth_m.shape != rgb.shape[:2]:
            raise RuntimeError(
                "Aligned RGB/depth shapes differ: "
                f"RGB={rgb.shape}, depth={depth_m.shape}"
            )

        result = CameraData(
            images={
                "left_rgb": np.ascontiguousarray(rgb)
            },
            timestamp=float(color_frame.get_timestamp()),
        )

        # CameraNode already knows how to forward this dynamic field.
        result.depth_data = np.ascontiguousarray(depth_m)

        return result
