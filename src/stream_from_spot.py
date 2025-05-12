"""Stream images from Spot and save them to file."""

import argparse
import pickle
import time
from pathlib import Path

from bosdyn.client.frame_helpers import get_a_tform_b

from src.spot_image_client import CAMERA_TO_OPERATING_RANGE_M, ImageFormat
from src.spot_manager import SpotManager
from transform_utils.kinematics import Point3D, Pose3D, Quaternion


def main() -> None:
    """Stream images from Spot and save them to file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("hostname", type=str, help="IP of the Spot robot")
    parser.add_argument("username", type=str, help="Username to authenticate with Spot")
    parser.add_argument("password", type=str, help="Password to authenticate with Spot")
    parser.add_argument("output_path", type=Path, help="Folder to output images into")
    parser.add_argument("overwrite", type=bool, help="Permit overwriting the output path")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    overwrite = args.overwrite
    assert overwrite or not output_path.exists(), f"Cannot overwrite existing path: {output_path}"

    images_folder = output_path / "images"
    poses_folder = output_path / "poses"
    for path in [output_path, images_folder, poses_folder]:
        path.mkdir(parents=True, exist_ok=True)

    manager = SpotManager("manager", args.hostname, args.username, args.password)
    manager.log_info("SpotManager now initialized...")

    cameras = ["hand", "frontleft", "frontright", "left", "right", "back"]
    formats = [ImageFormat.RGB, ImageFormat.DEPTH]
    request_types = [(c, f) for c in cameras for f in formats]
    requests = [manager.image_client.make_image_request(c, f) for (c, f) in request_types]

    assert None not in requests, "One of the image requests gave None."

    record_duration_s = 60

    end_time_t = time.time() + record_duration_s
    timestep_idx = 0
    while time.time() < end_time_t:
        responses = manager.image_client.get_images(requests)

        for req_type, response in zip(request_types, responses, strict=True):
            camera_name, img_format = req_type

            # Create a path to save this image (some format from some camera at timestep t)
            image_path = images_folder / str(img_format) / camera_name / f"{timestep_idx}.png"

            # pose_path = poses_folder / camera_name / f"{timestep_idx}.pkl"

            depth_scale = response.source.depth_scale if img_format == ImageFormat.DEPTH else None
            depth_range_m = (
                CAMERA_TO_OPERATING_RANGE_M["hand"]
                if camera_name == "hand"
                else CAMERA_TO_OPERATING_RANGE_M["body"]
            )

            manager.image_client.save_image_to_file(
                response,
                image_path,
                depth_scale,
                depth_range_m,
                None,
            )

            # camera_frame = response.shot.frame_name_image_sensor
            # tf_snapshot = response.shot.transforms_snapshot
            # tf_odom_camera = get_a_tform_b(tf_snapshot, camera_frame, "vision")
            # pose_odom_camera = Pose3D(
            #     Point3D(
            #         tf_odom_camera.position.x,
            #         tf_odom_camera.position.y,
            #         tf_odom_camera.position.z,
            #     ),
            #     Quaternion(
            #         w=tf_odom_camera.rotation.w,
            #         x=tf_odom_camera.rotation.x,
            #         y=tf_odom_camera.rotation.y,
            #         z=tf_odom_camera.rotation.z,
            #     ),
            # )

            # with pose_path.open("wb") as pose_f:
            #     print(pose_odom_camera)
            #     pickle.dump(pose_odom_camera, pose_f)

        timestep_idx += 1
        time.sleep(0.1)


if __name__ == "__main__":
    main()
