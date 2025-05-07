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
from transform_utils.kinematics import Point3D, Pose3D, Quaternion


@dataclass(frozen=True)
class FrameID:
    """An identifier for a frame based on its waypoint and index for that waypoint."""

    waypoint_id: str
    frame_idx: int  # Index of the frame for the waypoint (0 through 3)

    @classmethod
    def from_key(cls, key: str) -> FrameID | None:
        """Construct a FrameID based on a frame's string key.

        :param key: String identifying the frame (of form WAYPOINT_ID==-IDX)
        :return: Constructed FrameID instance, or None if invalid key format
        """
        parts = str(key).split("==-")
        if len(parts) != 2:
            print(f"Unexpected frame key format: {key}")
            return None

        return FrameID(parts[0], int(parts[1]))


@dataclass(frozen=True)
class PosedImage:
    """An image taken at a known camera pose."""

    image: np.ndarray
    pose: Pose3D


@dataclass(frozen=True)
class PosedRGBD:
    """An RGB-D image pair taken at a known camera pose."""

    rgb: np.ndarray
    depth: np.ndarray
    pose: Pose3D


def load_frame_poses(pose_pkl: Path) -> dict[FrameID, Pose3D]:
    """Load per-frame pose data from the given pickle file.

    :param pose_pkl: Filepath to a .pkl file containing pose data
    :return: Map from frame identifiers to corresponding poses
    """
    assert pose_pkl.suffix == ".pkl", f"{pose_pkl} is not a pickle (.pkl) file."

    try:
        with pose_pkl.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Pose pickle data was {type(data)}, not a dictionary.")

    except Exception as exc:
        print(f"Error loading poses from pickle file {pose_pkl}: {exc}")

    loaded_poses = {}
    for frame_key, pose_data in data.items():
        frame_id = FrameID.from_key(frame_key)
        if frame_id is None:
            print(f"Could not parse frame key: {frame_key}")
            continue

        x, y, z = pose_data["position"]
        qw, qx, qy, qz = pose_data["quaternion(wxyz)"]
        pose = Pose3D(Point3D(x, y, z), Quaternion(qx, qy, qz, qw))

        loaded_poses[frame_id] = pose

    return loaded_poses


def load_images(folder: Path) -> dict[FrameID, dict[str, np.ndarray]]:
    """Load RGB-D images from the given folder.

    :param folder: Folder containing RGB-D images collected on the robot
    :return: Map from frame IDs to maps from image types (as strings) to corresponding images
    """
    prefixes = ("color", "combined", "depth")

    prefixed_files = [fp for fp in folder.iterdir() if any(fp.name.startswith(p) for p in prefixes)]
    if not prefixed_files:
        raise RuntimeError(f"Didn't find any images in folder {folder} with prefixes {prefixes}!")

    frame_groups_paths: dict[FrameID, dict[str, Path]] = {}
    for filepath in prefixed_files:
        filename = filepath.name
        filename = filename.removesuffix(".jpg")

        for prefix in prefixes:
            if str(filename).startswith(prefix):
                frame_key = filename[len(prefix) + 1 :]
                frame_id = FrameID.from_key(frame_key)
                if frame_id is None:
                    print(f"Could not parse frame key: {frame_key}")
                    continue

                frame_groups_paths.setdefault(frame_id, {})[prefix] = filepath
                break

    frame_groups: dict[FrameID, dict[str, np.ndarray]] = {}
    for frame_id, image_paths_map in frame_groups_paths.items():
        print(f"Loading images for frame {frame_id.frame_idx} of waypoint {frame_id.waypoint_id}:")
        for image_type, path in image_paths_map.items():
            try:
                # Load depth images as pickled NumPy arrays
                if image_type == "depth":
                    with path.open("rb") as f:
                        arr = pickle.load(f)

                else:  # Load RGB and 'combined' images as images
                    img = Image.open(path)
                    arr = np.array(img)

            except Exception as exc:
                print(f"    Error reading {path.name}: {exc}")
            else:
                print(f"    {image_type} -> dtype: {arr.dtype}, shape: {arr.shape}")
                frame_groups.setdefault(frame_id, {})[image_type] = arr

    return frame_groups


@dataclass(frozen=True)
class CameraIntrinsics:
    """Intrinsic parameters for a pinhole model camera.

    Reference: https://ksimek.github.io/2013/08/13/intrinsic/

    Definitions:
        - Principal axis - Line perpendicular to the image plane through the camera pinhole.
        - Principle point - Where the principal axis intersects with the image plane, relative
            to the origin of the film (i.e., the pinhole's location if projected onto the film).

    """

    fx: float  # Focal length (pixels) in x
    fy: float  # Focal length (pixels) in y
    x0: float  # Principal point offset in x
    y0: float  # Principal point offset in y

    def to_list(self) -> list[float]:
        """Convert the camera intrinsics into a list: [fx, fy, x0, y0]."""
        return [self.fx, self.fy, self.x0, self.y0]

    def to_matrix(self) -> np.ndarray:
        """Convert the camera intrinsic parameters into a 3x3 intrinsic matrix."""
        return np.array([[self.fx, 0.0, self.x0], [0.0, self.fy, self.y0], [0.0, 0.0, 1.0]])


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

    height, width, channels = rgbd_data[0].rgb.shape
    assert width > height, "Expected images to be wider than taller."
    assert channels == 3, f"Expected RGB images to have 3 channels; found {channels}."

    manifest["w"] = width
    manifest["h"] = height

    manifest["integer_depth_scale"] = depth_scale / MAX_UINT16

    rgb_dir = output_path.joinpath("rgb")
    depth_dir = output_path.joinpath("depth")

    for path in [output_path, rgb_dir, depth_dir]:
        path.mkdir(parents=True)

    for frame_num, posed_rgbd in enumerate(rgbd_data):
        rgb = posed_rgbd.rgb.reshape((height, width, 3))
        rgb_filepath = rgb_dir.joinpath(f"{frame_num}.png")
        cv2.imwrite(str(rgb_filepath), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        assert width * height == posed_rgbd.depth.size, "Depth and RGB image sizes should match."

        depth = posed_rgbd.depth.reshape(height, width)
        depth_uint16 = ((depth / depth_scale) * MAX_UINT16).astype(np.uint16)
        depth_filepath = depth_dir.joinpath(f"{frame_num}.png")
        cv2.imwrite(str(depth_filepath), depth_uint16)

        # TODO: Should this matrix be transposed or not? Probably not (looked far worse when was)
        transform_w_c = posed_rgbd.pose.to_homogeneous_matrix()  # .view(dtype=np.float32)

        frame = {
            "transform_matrix": transform_w_c.tolist(),
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


def depthmap_to_pointmap(depthmap: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Convert a depthmap into a 3D pointmap in the world frame.

    Reference: https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html.
        See the equations in the description of the create_from_rgbd_image() function.

    :param depthmap: Depth map of shape (H, W)
    :param intrinsics: Camera intrinsic parameters
    :return: Array of 3D coordinates (i.e., points) corresponding to each input pixel (H, W, 3)
    """
    rows, cols = depthmap.shape
    U = np.tile(np.arange(rows).reshape(rows, 1), (1, cols))  # noqa: N806
    V = np.tile(np.arange(cols).reshape(1, cols), (rows, 1))  # noqa: N806
    Z = depthmap  # noqa: N806
    X = (U - intrinsics.x0) * Z / intrinsics.fx  # noqa: N806
    Y = (V - intrinsics.y0) * Z / intrinsics.fy  # noqa: N806

    return np.stack([X, Y, Z], axis=-1)  # (H, W, 3)


def main() -> None:
    """Load RGB-D data collected on the Spot robot from file."""
    ### DEBUGGING ###
    depth_path = Path(
        Path.home() / "Documents/SplaTAM/spot_room/depth_wedded-cocoon-E2.IPfP0By1xkWUdkwfVbg==-1",
    )
    with depth_path.open("rb") as f:
        depthmap = pickle.load(f)

    pointmap = depthmap_to_pointmap(depthmap, SPOT_RGB_HAND_CAMERA_INTRINSICS)

    zero_depth_mask = depthmap > 0
    points = pointmap[zero_depth_mask]  # Filter out zero-depth pixels
    print(f"Points with non-zero depth: {np.count_nonzero(points)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("output.ply", pcd)

    return
    parser = argparse.ArgumentParser(description="Import robot-collected RGB-D data from file.")
    parser.add_argument("image_folder", type=Path, help="Path to the folder containing images")
    parser.add_argument("poses_pkl", type=Path, help="Path to a pickle file containing poses")
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to which SplaTAM-compatible dataset will be written",
    )

    args = parser.parse_args()
    image_folder: Path = args.image_folder
    assert image_folder.exists(), f"Folder {image_folder} does not exist."
    poses_pkl: Path = args.poses_pkl
    assert poses_pkl.exists(), f"Poses pickle file {poses_pkl} does not exist."
    output_path: Path = args.output_path

    frame_poses = load_frame_poses(poses_pkl)
    loaded_images = load_images(image_folder)

    rgbd_dataset: list[PosedRGBD] = []
    for frame_id, images_map in loaded_images.items():
        if not {"color", "depth"}.issubset(images_map.keys()):
            print(f"Frame {frame_id} is missing either an RGB or depth image: {images_map}")
            continue

        if frame_id not in frame_poses:
            print(f"Frame {frame_id} is missing a pose!")
            continue

        rgb_image = images_map["color"]
        depth_image = images_map["depth"]
        pose = frame_poses[frame_id]

        rgbd_dataset.append(PosedRGBD(rgb_image, depth_image, pose))

    print(f"{len(rgbd_dataset)} RGB-D image pairs with poses have been imported from file.")

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
