"""Import RGB-D data collected on the Spot robot from file."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from src.vision_utils import CameraIntrinsics, PosedRGBD, depthmap_to_points
from transform_utils.kinematics import Point3D, Pose3D, Quaternion


@dataclass(frozen=True)
class CameraFrame:
    """Represents the pose of a camera at a particular frame in a trajectory."""

    frame_idx: int  # Index of the frame in the overall sequence
    camera: str  # Name of the camera
    pose_w_c: Pose3D  # Pose of the camera during the frame (camera w.r.t. world)


def load_camera_frames(poses_folder: Path) -> dict[tuple[int, str], CameraFrame]:
    """Load per-camera and per-frame pose data from the given folder.

    :param poses_folder: Path to a folder containing pickled poses
    :return: Map from (frame index, camera name) tuples to CameraFrame data structures
    """
    assert poses_folder.exists(), f"Cannot import poses from nonexistent folder: {poses_folder}"
    assert poses_folder.is_dir(), f"{poses_folder} is not a directory."

    output_map: dict[tuple[int, str], CameraFrame] = {}
    for pose_pkl in poses_folder.iterdir():
        if pose_pkl.suffix != ".pkl":
            continue

        pieces = pose_pkl.stem.split("-")
        if len(pieces) != 2:
            print(f"Could not parse pickled pose filename: {pose_pkl}")
            continue
        camera_name, frame_idx = pieces

        try:
            with pose_pkl.open("rb") as f:
                data = pickle.load(f)

            if not isinstance(data, Pose3D):
                raise TypeError(f"Imported pose had type {type(data)}, not Pose3D.")

            output_map[(frame_idx, camera_name)] = CameraFrame(frame_idx, camera_name, data)
        except Exception as exc:
            print(f"Error loading pose from pickle file {pose_pkl}: {exc}")

    return output_map


@dataclass
class RGBDPair:
    """A paired RGB image and depth image taken at the same time on the same camera."""

    frame_idx: int  # Index of the frame in the overall sequence
    camera: str  # Name of the camera used to take the images
    rgb: np.array  # RGB image
    depth: np.array  # Depth image


def load_images(folder: Path) -> dict[tuple[int, str], RGBDPair]:
    """Load RGB-D images from the given folder.

    :param folder: Folder containing RGB-D images collected on the robot
    :return: Map from (frame index, camera name) tuples to RGB-D image pairs
    """
    rgb_folder = folder / "ImageFormat.RGB"
    depth_folder = folder / "ImageFormat.DEPTH"

    rgb_count = sum(1 for i in rgb_folder.iterdir() if i.is_file())
    depth_count = sum(1 for i in depth_folder.iterdir() if i.is_file())
    assert rgb_count == depth_count, f"Found {rgb_count} RGB images but {depth_count} depth images."

    loaded_images: dict[tuple[int, str], RGBDPair] = {}
    for rgb_filepath in rgb_folder.iterdir():
        depth_filepath = depth_folder / rgb_filepath.name
        assert depth_filepath.exists(), f"Expected to find file named: {depth_filepath}"

        pieces = rgb_filepath.stem.split("-")
        if len(pieces) != 2:
            print(f"Could not parse RGB image filename: {rgb_filepath}")
            continue
        camera_name, frame_idx = pieces

        rgb_image = Image.open(rgb_filepath)
        rgb_arr = np.array(rgb_image)
        depth_image = Image.open(depth_filepath)
        depth_arr = np.array(depth_image)

        rgbd_pair = RGBDPair(frame_idx, camera_name, rgb_arr, depth_arr)
        loaded_images[(frame_idx, camera_name)] = rgbd_pair

    return loaded_images


SPOT_RGB_HAND_CAMERA_INTRINSICS = CameraIntrinsics(
    fx=552.0291012161067,
    fy=552.0291012161067,
    x0=320,
    y0=240,
)


def load_camera_intrinsics() -> CameraIntrinsics:
    """Load Spot's RGB hand camera intrinsics from file.

    TODO: Actually collect these from the robot!
    """
    return SPOT_RGB_HAND_CAMERA_INTRINSICS


MAX_UINT16 = 65535.0  # Maximum value of a 16-bit unsigned integer


def create_manifest(
    rgbd_data: list[PosedRGBD],
    intrinsics: CameraIntrinsics,
    output_path: Path,
    depth_scale: float,
) -> dict[str, Any]:
    """Create a manifest as exported by SplaTAM's nerfcapture2dataset.py script.

    :param rgbd_data: Dataset of posed RGB-D image pairs
    :param intrinsics: Camera intrinsics for the camera collecting the images
    :param output_path: Path to which data will be output
    :param depth_scale: Maximum value of depth data; used for scaling
    """
    assert rgbd_data, "Cannot create a manifest for an empty RGB-D dataset!"
    if output_path.exists():
        raise FileExistsError(f"Cannot overwrite existing directory: {output_path}")

    fx, fy, x0, y0 = intrinsics.to_list()
    manifest = {"fl_x": fx, "fl_y": fy, "cx": x0, "cy": y0, "frames": []}

    height, width, _ = rgbd_data[0].rgb.shape
    manifest["w"] = width
    manifest["h"] = height

    manifest["integer_depth_scale"] = depth_scale / MAX_UINT16

    rgb_dir = output_path.joinpath("rgb")
    depth_dir = output_path.joinpath("depth")

    for path in [output_path, rgb_dir, depth_dir]:
        path.mkdir(parents=True)

    for frame_num, posed_rgbd in enumerate(rgbd_data):
        rgb = posed_rgbd.rgb.reshape(height, width, 3)
        rgb_filepath = rgb_dir.joinpath(f"{frame_num}.png")
        cv2.imwrite(str(rgb_filepath), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        assert width * height == posed_rgbd.depth.size, "Depth and RGB image sizes should match."

        depth = posed_rgbd.depth.reshape(height, width)
        depth_uint16 = ((depth / depth_scale) * MAX_UINT16).astype(np.uint16)
        depth_filepath = depth_dir.joinpath(f"{frame_num}.png")
        cv2.imwrite(str(depth_filepath), depth_uint16)

        frame = {
            "transform_matrix": posed_rgbd.pose_w_c.to_homogeneous_matrix().tolist(),
            "file_path": str(rgb_filepath),
            "fl_x": fx,
            "fl_y": fy,
            "cx": x0,
            "cy": y0,
            "w": width,
            "h": height,
            "depth_path": str(depth_filepath),
        }
        manifest["frames"].append(frame)

    return manifest


def main() -> None:
    """Load RGB-D data collected on the Spot robot from file."""
    parser = argparse.ArgumentParser(description="Import robot-collected RGB-D data from file.")
    parser.add_argument("input_path", type=Path, help="Path to the folder containing data")
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to which SplaTAM-compatible dataset will be written",
    )

    args = parser.parse_args()
    input_path: Path = args.input_path
    assert input_path.exists(), f"Folder {input_path} does not exist."
    output_path: Path = args.output_path

    camera_frames = load_camera_frames(input_path / "poses")
    images = load_images(input_path / "images")

    rgbd_dataset: list[PosedRGBD] = []
    for idx_tuple, frame_data in camera_frames.items():
        if idx_tuple not in images:
            print(f"Frame {idx_tuple[0]} for camera {idx_tuple[1]} doesn't have an RGB-D pair!")
            continue

        rgbd = images[idx_tuple]
        print(f"RGB image shape: {rgbd.rgb.shape}   Depth image shape: {rgbd.depth.shape}")

        rgbd_dataset.append(PosedRGBD(rgbd.rgb, rgbd.depth, frame_data.pose_w_c))

    print(f"{len(rgbd_dataset)} RGB-D image pairs with poses have been imported from file.")

    # Visualize the pointcloud resulting from the imported depth images and poses
    list_points_w = []  # w.r.t. world frame
    list_colors = []
    for posed_rgbd in rgbd_dataset:
        print(posed_rgbd.pose_w_c)
        translation_w_c = posed_rgbd.pose_w_c.position.to_array()  # (3,)
        rotation_w_c = posed_rgbd.pose_w_c.orientation.to_rotation_matrix()  # (3, 3)

        points_c = depthmap_to_points(posed_rgbd.depth, SPOT_RGB_HAND_CAMERA_INTRINSICS)  # (N, 3)
        points_w = (rotation_w_c @ points_c.T).T + translation_w_c  # Still (N, 3)

        colors = posed_rgbd.rgb[posed_rgbd.depth > 0]  # Also (N, 3)

        list_points_w.append(points_w)
        list_colors.append(colors)

    all_points_w = np.concatenate(list_points_w, axis=0)
    all_colors = np.concatenate(list_colors, axis=0) / 255.0
    print(np.min(all_colors), np.max(all_colors))
    print(f"Imported {all_points_w.shape[0]} points from RGB-D images.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points_w)
    # pcd.colors = o3d.utility.Vector3dVector(all_colors)
    o3d.visualization.draw_geometries([pcd])

    input("Press enter to output the JSON manifest for SplaTAM...")

    # Calculate the depth scale seemingly used by this RGB-D dataset
    min_depth_value = min(np.min(posed_rgbd.depth) for posed_rgbd in rgbd_dataset)
    max_depth_value = max(np.max(posed_rgbd.depth) for posed_rgbd in rgbd_dataset)
    print(f"Minimum observed depth value: {min_depth_value}")
    print(f"Maximum observed depth value: {max_depth_value}")
    depth_scale = 10.0  # Reasonable maximum value (m) for depth data from Spot's hand camera

    manifest = create_manifest(rgbd_dataset, load_camera_intrinsics(), output_path, depth_scale)

    # Write the manifest to JSON
    manifest_json = json.dumps(manifest, indent=4)
    with output_path.joinpath("transforms.json").open("w") as f:
        f.write(manifest_json)


if __name__ == "__main__":
    main()
