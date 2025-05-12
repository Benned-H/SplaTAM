"""Output data collected from the Spot robot into the COLMAP capture format."""

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import liblzfse
import numpy as np
from PIL import Image

from src.vision_utils import CameraIntrinsics, PosedRGBD

MAX_UINT16 = 65535.0  # Maximum value of a 16-bit unsigned integer


def export_colmap_text(
    rgbd_data: list[PosedRGBD],
    intrinsics_per_camera: dict[str, CameraIntrinsics],
    output_path: Path,
) -> None:
    """Export an RGB-D dataset to the three-text-file COLMAP capture format.

    :param rgbd_data: List of posed RGB-D pairs (RGB, depth, and pose_w_c)
    :param intrinsics_per_camera: Dictionary mapping camera names to their intrinsic parameters
    :param output_path: Directory within which the COLMAP capture is output
    """
    assert rgbd_data, "Cannot create a COLMAP capture with an empty RGB-D dataset!"
    if output_path.exists():
        raise FileExistsError(f"Cannot overwrite existing directory: {output_path}")

    camera_names = list(intrinsics_per_camera.keys())
    camera_id_map = {name: idx + 1 for idx, name in enumerate(camera_names)}

    # Make output paths
    rgb_root = output_path / "rgb"
    depth_root = output_path / "depth"
    for d in (output_path, rgb_root, depth_root):
        d.mkdir(parents=True, exist_ok=False)
    for c in camera_names:
        (rgb_root / c).mkdir()
        (depth_root / c).mkdir()

    # Save PNGs and record per-camera resolution
    camera_resolutions: dict[str, tuple[int, int]] = {}
    fx_fy_cx_cy: dict[str, tuple[float, float, float, float]] = {}
    for idx, rgbd_pair in enumerate(rgbd_data):
        camera = "hand"  # TODO: Should be = rgbd_pair.camera_name
        intrinsics = intrinsics_per_camera[camera]
        fx_fy_cx_cy[camera] = tuple(intrinsics.to_list())

        height, width, _ = rgbd_pair.rgb.shape
        camera_resolutions.setdefault(camera, (height, width))

        # Write RGB image to file
        rgb_path = rgb_root / camera / f"{idx:06d}.png"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgbd_pair.rgb, cv2.COLOR_RGB2BGR))

        # Write depth image to file
        depth_scale = 1.0  # TODO: argument!
        depth = rgbd_pair.depth.reshape(height, width)
        depth16 = ((depth / depth_scale) * MAX_UINT16).astype(np.uint16)
        depth_path = depth_root / camera / f"{idx:06d}.png"
        cv2.imwrite(str(depth_path), depth16)

    # Write cameras.txt
    camera_model = "pinhole"  # TODO: Where from?
    with (output_path / "cameras.txt").open("w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(camera_names)}\n")
        for camera in camera_names:
            idx = camera_id_map[camera]
            height, width = camera_resolutions[camera]
            params = fx_fy_cx_cy[camera]
            params_str = " ".join(str(x) for x in params)
            f.write(f"{idx} {camera_model} {width} {height} {params_str}\n")

    # Write images.txt
    with (output_path / "images.txt").open("w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(rgbd_data)}\n")
        for idx, frame in enumerate(rgbd_data):
            image_id = idx + 1
            camera_name = "hand"  # TODO: camera_id_map[frame.camera_name]
            camera_id = camera_id_map[camera_name]

            q = frame.pose_w_c.orientation.to_array()
            qw, qx, qy, qz = q[3], q[0], q[1], q[2]  # Quaternion -> qw, qx, qy, qz
            pos = frame.pose_w_c.position
            tx, ty, tz = pos.x, pos.y, pos.z  # Translation

            name = f"{camera_name}/{idx:06d}.png"

            f.write(
                f"{image_id} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} "
                f"{tx:.6f} {ty:.6f} {tz:.6f} {camera_id} {name}\n\n",
            )


def export_record3d_folder(
    rgbd_data: list[PosedRGBD],
    intrinsics: CameraIntrinsics,
    output_folder: Path,
    *,
    timestamps: list[float] | None = None,
    init_pose: list[float] | None = None,
    icon_path: Path | None = None,
    audio_path: Path | None = None,
    # If you still want LZFSE‐compressed depth/conf, set this True:
    use_lzfse: bool = False,
):
    """Write a Record3D‐style folder (not zipped) with:
      color/<i>.jpg
      depth/<i>.exr       (or .depth if use_lzfse=True)
      metadata            (JSON)
      [icon], [sound.m4a]

    After running this, point Nerfstudio at `--data output_folder`.

    Parameters
    ----------
    rgbd_data
        List of PosedRGBD, each with `.rgb` (H×W×3 uint8), `.depth` (H×W float32),
        and `.pose_w_c` (Pose3D: .orientation → [x,y,z,w], .position → [x,y,z]).
    intrinsics
        CameraIntrinsics, whose `.to_list()` → [fx, fy, cx, cy].
    output_folder
        Directory to create (must not already exist).
    timestamps
        Optional list of floats, one per frame.
    init_pose
        Optional 7‐element [qx,qy,qz,qw,x,y,z].
    icon_path
        Optional Path to a JPEG thumbnail to copy as `icon`.
    audio_path
        Optional Path to an M4A audio track to copy as `sound.m4a`.
    use_lzfse
        If True, writes depth/conf as LZFSE-compressed `.depth`/`.conf` instead
        of raw EXR. Default False (EXR).

    """
    """Write a .r3d Record3D file from RGB-D frames + poses.

    - rgbd_data[i].rgb: H×W×3 uint8
    - rgbd_data[i].depth: H×W float32 (meters)
    - rgbd_data[i].pose_w_c: Pose3D (world→camera)
    """
    if output_folder.exists():
        raise FileExistsError(f"{output_folder} already exists")
    # create folders
    color_dir = output_folder / "color"
    depth_dir = output_folder / "depth"
    color_dir.mkdir(parents=True)
    depth_dir.mkdir()

    # unpack intrinsics
    fx, fy, cx, cy = intrinsics.to_list()

    # gather metadata lists
    poses: list[list[float]] = []
    frame_ts = timestamps or []

    for i, frame in enumerate(rgbd_data):
        H, W, _ = frame.rgb.shape

        # 1) color → JPEG
        img = Image.fromarray(frame.rgb.reshape(H, W, 3))
        img.save(color_dir / f"{i}.jpg")

        # 2) depth → EXR (or LZFSE .depth)
        depth = frame.depth.astype(np.float32)
        if use_lzfse:
            import pylzfse

            raw = depth.tobytes()
            comp = pylzfse.compress(raw)
            (depth_dir / f"{i}.depth").write_bytes(comp)
            # optional confidence:
            # conf = np.ones((H, W), np.uint8)
            # rawc = conf.tobytes()
            # compc = pylzfse.compress(rawc)
            # (depth_dir / f"{i}.conf").write_bytes(compc)
        else:
            # write raw EXR
            import imageio

            imageio.imwrite(str(depth_dir / f"{i}.exr"), depth)

        # 3) poses
        q = frame.pose_w_c.orientation.to_array()
        qw, qx, qy, qz = q[3], q[0], q[1], q[2]  # Quaternion -> qw, qx, qy, qz
        pos = frame.pose_w_c.position
        tx, ty, tz = pos.x, pos.y, pos.z  # Translation

        poses.append([qw, qx, qy, qz, tx, ty, tz])

    # 4) copy icon/audio if provided
    if icon_path:
        (output_folder / "icon").write_bytes(icon_path.read_bytes())
    if audio_path:
        (output_folder / "sound.m4a").write_bytes(audio_path.read_bytes())

    # 5) write metadata
    meta: dict[str, Any] = {
        "version": "1.10.3",
        "width": W,
        "height": H,
        "intrinsics": [fx, fy, cx, cy],
        "poses": poses,
        "frameTimestamps": frame_ts,
    }
    if init_pose:
        meta["initPose"] = init_pose

    # Record3D sometimes expects the file named just `metadata`
    (output_folder / "metadata").write_text(
        json.dumps(meta, separators=(",", ":"), ensure_ascii=False),
    )
