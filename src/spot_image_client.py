"""Define a class providing utility function access to the Spot SDK image client."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from os import devnull
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from bosdyn.api.image_pb2 import Image, ImageRequest, ImageResponse
from bosdyn.client.image import ImageClient, build_image_request

from transform_utils.logging import log_error, log_info

if TYPE_CHECKING:
    from bosdyn.client.robot import Robot

from contextlib import redirect_stdout


@dataclass(frozen=True)
class PixelFormatSpec:
    """Static mapping from Spot SDK pixel formats to sensor_msgs/Image specifications.

    Reference: https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html
    """

    encoding: str  # ROS image encoding string (see image_encodings.h)
    dtype: type[np.generic]  # NumPy datatype used when reshaping raw pixel data
    channels: int  # Number of channels in the raw buffer (e.g., 3 for RGB)


PIXEL_FORMAT_SPECS: dict[int, PixelFormatSpec] = {
    Image.PIXEL_FORMAT_RGB_U8: PixelFormatSpec("bgr8", np.uint8, 3),  # cv_bridge expects BGR
    Image.PIXEL_FORMAT_RGBA_U8: PixelFormatSpec("bgra8", np.uint8, 4),
    Image.PIXEL_FORMAT_GREYSCALE_U8: PixelFormatSpec("mono8", np.uint8, 1),
    Image.PIXEL_FORMAT_GREYSCALE_U16: PixelFormatSpec("mono16", np.uint16, 1),
    Image.PIXEL_FORMAT_DEPTH_U16: PixelFormatSpec("16UC1", np.uint16, 1),
}
DEFAULT_PIXEL_SPEC = PixelFormatSpec("mono8", np.uint8, 1)


class ImageFormat(Enum):
    """Subset of pixel formats available from Spot that map cleanly to ROS encodings.

    Reference:
        https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#image-pixelformat
    """

    RGB = 1
    GREYSCALE = 2
    DEPTH = 3

    def pixel_format(self) -> int:
        """Return the corresponding Spot SDK pixel format constant."""
        return {
            ImageFormat.RGB: Image.PIXEL_FORMAT_RGB_U8,  # Three bytes per pixel
            ImageFormat.GREYSCALE: Image.PIXEL_FORMAT_GREYSCALE_U8,  # One byte per pixel
            ImageFormat.DEPTH: Image.PIXEL_FORMAT_DEPTH_U16,  # z-distance from camera (mm)
        }[self]

    def __repr__(self):
        return {
            ImageFormat.RGB: "rgb",
            ImageFormat.GREYSCALE: "greyscale",
            ImageFormat.DEPTH: "depth",
        }[self]


CAMERA_TO_OPERATING_RANGE_M = {
    "body": (0.3, 4.0),  # 0.3 m up to 2-4 meters
    "hand": (0.4, 6.0),  # 0.4 m up to at least 6 meters
}


class SpotImageClient:
    """A wrapper providing utility function access to Spot's image client."""

    def __init__(self, robot: Robot, debug_mode: bool) -> None:
        """Initialize an image client using the given robot.

        :param robot: Point of access for Spot's RPC clients
        :param debug_mode: True if debug messages and images should be published, else False
        """
        self._image_client = robot.ensure_client(ImageClient.default_service_name)

        self.debug_mode = debug_mode

        # Identify the image sources available from Spot
        image_sources_proto = self._image_client.list_image_sources()
        self.image_sources = [source.name for source in image_sources_proto]
        self.camera_names = ["frontleft", "frontright", "left", "right", "back", "hand"]

        # if debug_mode:
        #     self.run_diagnostics()

    def make_image_request(self, camera: str, image_format: ImageFormat) -> ImageRequest | None:
        """Build an image request Protobuf message to be sent to Spot.

        :param camera: Name of the camera to be used to capture the image
        :param image_format: Format of image requested (e.g., RGB or DEPTH)
        :return: Image request Protobuf message, or None if invalid inputs given
        """
        source = self._camera_to_image_source(camera, image_format)

        if source not in self.image_sources:
            log_error(f"Unrecognized image source: '{source}'")
            return None

        return build_image_request(source, pixel_format=image_format.pixel_format())

    def get_images(self, requests: list[ImageRequest]) -> list[ImageResponse]:
        """Request a collection of images from the robot.

        Reference: https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/image.html

        :param requests: List of images requested from the robot
        :return: List of resulting image responses from Spot
        """
        responses = self._image_client.get_image(requests)

        if len(responses) != len(requests):
            log_error(f"Expected {len(requests)} responses, but received {len(responses)}.")
            return []

        return responses

    def get_rgbd_pair(self, camera: str) -> list[ImageResponse]:
        """Request a pair of RGB-D images from the given camera on Spot.

        :param camera: Name of the camera to be used to capture the image (e.g., "right" or "hand")
        :return: Tuple of (RGB, depth) image response Protobuf messages
        """
        requests = [
            self.make_image_request(camera, ImageFormat.RGB),
            self.make_image_request(camera, ImageFormat.DEPTH),
        ]

        responses = self.get_images(requests)
        assert len(responses) == len(requests)
        return responses

    def save_image_to_file(
        self,
        image: ImageResponse | np.ndarray,
        output_path: str | Path,
        depth_scale: float | None,
        depth_range_m: tuple[int, int],
        colormap: int | None = cv2.COLORMAP_MAGMA,
    ) -> None:
        """Save the given image (response, message, or data) to the file system.

        :param image: Image response/message/data to be exported to file
        :param output_path: Output path for the image (.png is appended if missing)
        :param depth_scale: Pixel value corresponding to a depth of 1 m (None for non-depth images)
        :param depth_range_m: Range of depths (in m) output from Spot's depth cameras
        :param colormap: Colormap used to visualize depth frames (None means no colorization)
        """
        outfile = Path(output_path).with_suffix(".png")
        outfile.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(image, ImageResponse):
            np_data = self.proto_to_np(image)
            is_depth = image.shot.image.pixel_format == Image.PIXEL_FORMAT_DEPTH_U16
        elif isinstance(image, np.ndarray):
            np_data = image
            is_depth = image.dtype == np.uint16
        else:
            log_error(f"[save_image_to_file] Unsupported image type: {type(image).__name__}")
            return

        # print(f"Depth scale: {depth_scale}")
        # if is_depth:  # Scale depth images to express depth in meters
        #     assert depth_scale is not None, "Depth images require a depth scale value."
        #     np_data = np_data / depth_scale

        # Colorize depth images, if requested
        if colormap is not None and is_depth:
            np_data = np.squeeze(np_data)  # Ensure 2D before masking

            valid = np_data > 0  # Ignore invalid pixels when scaling
            if np.any(valid):
                # near_mm, far_mm = depth_range_mm
                image8 = np.zeros_like(np_data, dtype=np.uint8)

                # clipped = np.clip(np_data[valid], near_mm, far_mm)
                scaled = cv2.normalize(np_data, None, 0, 255, cv2.NORM_MINMAX)
                scaled_flat = scaled.astype(np.uint8).ravel()

                image8[valid] = scaled_flat
                np_data = cv2.applyColorMap(image8, colormap)

            # near_m, far_m = depth_range_m  # Ignore invalid pixels when scaling
            # valid = np.logical_and(np_data > near_m, np_data < far_m)

        # Write the image to file
        success = cv2.imwrite(str(outfile), np_data)
        if success and outfile.exists():
            log_info(f"[SpotImageClient] Wrote image to {outfile}")
        else:
            log_error(f"cv2.imwrite failed for {outfile}")

    def run_diagnostics(
        self,
        repub_hz: float = 2.0,
        duration_s: float = 3.0,
        out_dir: str | Path | None = None,
    ) -> None:
        """Stream every image source, check image shape math, re-publish images, and save PNGs.

        :param repub_hz: Frequency (Hz) of image republishing
        :param duration_s: Duration (seconds) to run diagnostics
        :param out_dir: Output directory for exported PNGs (if None, defaults to ~/spot_images)
        """
        out_dir = Path.home() / "spot_images" if out_dir is None else Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        formats = {
            "rgb": ImageFormat.RGB,
            "greyscale": ImageFormat.GREYSCALE,
            "depth": ImageFormat.DEPTH,
        }

        requests: list[ImageRequest] = [
            build_image_request(source, pixel_format=f.pixel_format())
            for source in self.image_sources
            for f in formats.values()
        ]
        labels: list[tuple[str, str]] = [
            (source, format_str) for source in self.image_sources for format_str in formats
        ]

        start = time.time()

        saved_pairs: set[tuple[str, str]] = set()  # Remember which PNGs we're written

        while (time.time() - start) < duration_s:
            for request, (source, fmt) in zip(requests, labels, strict=False):
                try:
                    response = self.get_images([request])[0]

                    # Save the first PNG for each (source, format) pair
                    key = (source, fmt)
                    if key not in saved_pairs:
                        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        outfile = out_dir / f"{source}_{fmt}_{now_str}.png"

                        if "hand" in source:
                            depth_range_mm = CAMERA_TO_OPERATING_RANGE_MM["hand"]
                        else:
                            depth_range_mm = CAMERA_TO_OPERATING_RANGE_MM["body"]

                        self.save_image_to_file(response, outfile, depth_range_mm)
                        saved_pairs.add(key)

                except Exception as exc:
                    log_error(
                        f"[SpotImageClient] Exception for format {fmt} from {source}: \n\t{exc}",
                    )

            time.sleep(0.1)

        log_info(f"[SpotImageClient] Diagnostics complete - Files saved in {out_dir}")

    @staticmethod
    def proto_to_np(response: ImageResponse) -> np.ndarray:
        """Convert an ImageResponse Protobuf message into a NumPy array of image data.

        :param response: Protobuf message containing an image response from Spot
        :return: NumPy array containing converted image data
        """
        image = response.shot.image
        rows, cols = image.rows, image.cols
        spec = PIXEL_FORMAT_SPECS.get(image.pixel_format, DEFAULT_PIXEL_SPEC)
        buffer = np.frombuffer(image.data, dtype=spec.dtype)

        if image.format == Image.FORMAT_RAW:
            try:
                shape = (rows, cols) if spec.channels == 1 else (rows, cols, spec.channels)
                arr = buffer.reshape(shape)
            except ValueError:
                # Typically caused by JPEG data being sent as RAW - let OpenCV handle it
                log_error("Raw reshape failed - falling back to cv2.imdecode")
                arr = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        else:
            arr = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)

        return np.squeeze(arr)

    def _camera_to_image_source(self, camera_name: str, image_format: ImageFormat) -> str:
        """Translate a human-friendly camera name to a Spot SDK image source.

        :param camera_name: Name of a camera on Spot (e.g., "frontright" or "back")
        :param image_format: Format of image requested (e.g., RGB or DEPTH)
        :return: Name of the corresponding Spot SDK image source
        """
        if camera_name not in self.camera_names:
            log_error(f"Unrecognized camera name: '{camera_name}'.")
            return ""

        if camera_name == "hand":
            return (
                "hand_depth_in_hand_color_frame"
                if image_format == ImageFormat.DEPTH
                else "hand_color_image"
            )

        suffix = "depth_in_visual_frame" if image_format == ImageFormat.DEPTH else "fisheye_image"
        return f"{camera_name}_{suffix}"
