"""Define general-purpose utility classes and functions for computer vision operations."""

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from transform_utils.kinematics import Pose3D


@dataclass(frozen=True)
class PosedRGBD:
    """An RGB-D image pair taken at a known camera pose."""

    rgb: np.ndarray  # Shape (H, W, 3)
    depth: np.ndarray  # Shape (H, W)
    pose_w_c: Pose3D  # Pose of the camera w.r.t. the world frame when the images were taken

    def __post_init__(self) -> None:
        """Verify the expected dimensions of the RGB and depth images."""
        assert len(self.rgb.shape) == 3, f"RGB image had {len(self.rgb.shape)} dimensions, not 3."
        assert len(self.depth.shape) == 2, (
            f"Depth image had {len(self.depth.shape)} dimensions, not 2."
        )

        assert self.rgb.shape[0] == self.depth.shape[0], "Expected matching image heights."
        assert self.rgb.shape[1] == self.depth.shape[1], "Expected matching image widths."
        assert self.rgb.shape[2] == 3, f"RGB image had {self.rgb.shape[2]} channels, not 3."


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


def depthmap_to_points(depthmap: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Convert a depthmap into 3D points in the world frame.

    Reference: https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html.
        See the equations in the description of the create_from_rgbd_image() function.

    :param depthmap: Depth map of shape (H, W)
    :param intrinsics: Camera intrinsic parameters
    :return: Array of 3D points (w.r.t. camera) corresponding to non-zero-depth pixels (N, 3)
    """
    height, width = depthmap.shape  # rows -> height, columns -> width
    V, U = np.indices(dimensions=(height, width))  # U for width (x) and V for height (y)
    Z = depthmap
    X = (U - intrinsics.x0) * Z / intrinsics.fx
    Y = (V - intrinsics.y0) * Z / intrinsics.fy

    pointmap = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)
    assert pointmap.shape == (height, width, 3), (
        f"Expected pointmap shape to be {(height, width, 3)}; found shape {pointmap.shape}."
    )

    return pointmap[depthmap > 0]  # Filter out zero-depth pixels -> Shape (N, 3)


def main() -> None:
    """Test the various computer vision utility classes and functions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-file", type=Path, help="Path to a single depth image file")
    parser.add_argument("--depth-folder", type=Path, help="Path to a folder of depth image files")

    SPOT_RGB_HAND_CAMERA_INTRINSICS = CameraIntrinsics(
        fx=552.0291012161067,
        fy=552.0291012161067,
        x0=320,
        y0=240,
    )

    args = parser.parse_args()
    if args.depth_file is not None:
        depth_file: Path = args.depth_file
        assert depth_file.exists() and depth_file.is_file(), (
            f"Expected depth image at path {depth_file}."
        )

        with depth_file.open("rb") as f:
            depthmap = pickle.load(f)

        pointmap = depthmap_to_pointmap(depthmap, SPOT_RGB_HAND_CAMERA_INTRINSICS)
        print(pointmap.shape)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointmap)
        o3d.visualization.draw_geometries([pcd])
        # o3d.io.write_point_cloud("output.ply", pcd)

    if args.depth_folder is not None:
        depth_folder: Path = args.depth_folder
        assert depth_folder.exists() and depth_folder.is_dir(), (
            f"Expected path {depth_folder} to be a folder containing depth images."
        )

        pointmaps = []
        for file in depth_folder.iterdir():
            if not file.name.startswith("depth"):
                continue

            with file.open("rb") as f:
                depthmap = pickle.load(f)

            pointmap = depthmap_to_points(depthmap, SPOT_RGB_HAND_CAMERA_INTRINSICS)
            pointmaps.append(pointmap)

        if not pointmaps:
            raise FileNotFoundError(f"Found no files starting with 'depth' in {depth_folder}.")

        combined_pointmap = np.concatenate(pointmaps, axis=0)
        print(f"Combined pointmap has {combined_pointmap.shape[0]} total points.")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_pointmap)
        o3d.visualization.draw_geometries([pcd])  # This works, but it's in the camera frame(s)


if __name__ == "__main__":
    main()
