#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import panda_py


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def make_camera(args):
    if args.camera == "zed":
        from robots_realtime.sensors.cameras.zed_camera import (
            ZedCamera,
        )

        return ZedCamera(
            device_id=args.device_id,
            resolution=args.resolution,
            fps=args.fps,
            enable_depth=True,
            extrinsics_file=args.extrinsics,
        )

    if args.camera == "realsense":
        from robots_realtime.sensors.cameras.capx_realsense_camera import (
            CapXRealSenseCamera,
        )

        return CapXRealSenseCamera(
            device_id=args.device_id,
            resolution=args.resolution,
            fps=args.fps,
            enable_depth=True,
            extrinsics_file=args.extrinsics,
        )

    raise ValueError(args.camera)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera",
        choices=["zed", "realsense"],
        required=True,
    )
    parser.add_argument(
        "--device-id",
        required=True,
    )
    parser.add_argument(
        "--extrinsics",
        required=True,
    )
    parser.add_argument(
        "--franka-ip",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--resolution",
        default=None,
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--gripper-fraction",
        type=float,
        required=True,
        help="0=closed, 1=open",
    )
    parser.add_argument(
        "--robot-quaternion",
        choices=["xyzw", "wxyz"],
        default="xyzw",
    )
    parser.add_argument(
        "--objects",
        nargs="*",
        default=[],
    )

    args = parser.parse_args()

    if not 0.0 <= args.gripper_fraction <= 1.0:
        raise ValueError(
            "--gripper-fraction must be in [0,1]"
        )

    if args.resolution is None:
        args.resolution = (
            "HD720"
            if args.camera == "zed"
            else "VGA"
        )

    if args.fps is None:
        args.fps = (
            15
            if args.camera == "zed"
            else 30
        )

    out = Path(args.output).expanduser()
    out.mkdir(parents=True, exist_ok=False)

    print("Opening camera...")
    cam = make_camera(args)

    try:
        data = cam.read()

        rgb = np.asarray(
            data.images["left_rgb"]
        )
        depth = np.asarray(
            data.depth_data
        )

        if rgb.shape[:2] != depth.shape:
            raise RuntimeError(
                f"RGB/depth mismatch: "
                f"{rgb.shape} vs {depth.shape}"
            )

        np.save(
            out / "initial_rgb.npy",
            rgb,
        )

        np.save(
            out / "initial_depth_m.npy",
            depth,
        )

        cv2.imwrite(
            str(out / "initial_rgb.png"),
            cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR,
            ),
        )

        with open(
            out / "camera_intrinsics.json",
            "w",
        ) as f:
            json.dump(
                jsonable(cam.intrinsic_data),
                f,
                indent=2,
            )

        with open(
            out / "camera_extrinsics.json",
            "w",
        ) as f:
            json.dump(
                jsonable(cam.extrinsics),
                f,
                indent=2,
            )

        shutil.copy2(
            args.extrinsics,
            out / "camera_extrinsics_used.yaml",
        )

    finally:
        cam.stop()

    print("Reading Franka state...")

    robot = panda_py.Panda(
        args.franka_ip
    )

    state = robot.get_state()

    q = np.asarray(
        state.q,
        dtype=np.float64,
    )

    xyz = np.asarray(
        robot.get_position(),
        dtype=np.float64,
    )

    quat = np.asarray(
        robot.get_orientation(
            scalar_first=(
                args.robot_quaternion == "wxyz"
            )
        ),
        dtype=np.float64,
    )

    g = float(
        args.gripper_fraction
    )

    robot_data = {
        "joint_positions_7": q.tolist(),
        "ee_position_xyz_m": xyz.tolist(),
        "ee_quaternion": quat.tolist(),
        "ee_quaternion_order": (
            args.robot_quaternion
        ),
        "gripper_fraction": g,
        "pose_matrix": np.asarray(
            robot.get_pose()
        ).tolist(),
    }

    with open(
        out / "robot_state.json",
        "w",
    ) as f:
        json.dump(
            robot_data,
            f,
            indent=2,
        )

    objects = {}

    for name in args.objects:
        objects[name] = {
            "position_xyz_m": [
                None,
                None,
                None,
            ],
            "quaternion_wxyz": [
                None,
                None,
                None,
                None,
            ],
        }

    gt = {
        "coordinate_frame": "robot_base",
        "objects": objects,
        "robot": {
            "cartesian_pose": (
                xyz.tolist()
                + quat.tolist()
                + [g]
            ),
            "joint_positions": (
                q.tolist()
                + [g]
            ),
        },
    }

    with open(
        out / "gt_snapshot_skeleton.json",
        "w",
    ) as f:
        json.dump(
            gt,
            f,
            indent=2,
        )

    print()
    print("Snapshot saved to:")
    print(out)
    print()
    print("IMPORTANT:")
    print(
        "Object poses are intentionally null. "
        "Fill them from independent GT."
    )


if __name__ == "__main__":
    main()
