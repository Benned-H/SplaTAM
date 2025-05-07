#!/usr/bin/env python3

"""Import collected data from file and introspect its properties."""

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


def introspect_pickle(filepath: Path) -> None:
    """Load a pickle file and print its type and value."""
    print(f"Pickle file: {filepath}")

    try:
        with filepath.open("rb") as f:
            data = pickle.load(f)
        print(f"    Type: {type(data)}")

        if hasattr(data, "shape"):  # If it's array-like, print its shape
            print(f"    Shape: {data.shape}")

        if isinstance(data, dict):  # If it's a dict, list its keys
            print(f"    Keys: {list(data.keys())}")
            for k, v in data.items():
                print(f"    Key: {k} Value: {v}")
        elif isinstance(data, (list, tuple)):  # If it's a list or tuple, print its length
            print(f"    Length: {len(data)}")
        else:  # Otherwise, print a small preview of the data
            preview = repr(data)
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"    Value preview: {preview}")
    except Exception as exc:
        print(f"    Error loading pickle: {exc}")


@dataclass(frozen=True)
class ImageData:
    """Identifying information for an image imported from file."""

    waypoint_id: str  # ID of the waypoint corresponding to this image
    image_type: str  # String describing the type of image (e.g., "color" or "depth")


def introspect_images(folder: Path) -> None:
    """Load images with prefixes 'color_', 'depth_', and 'combined_' and print their shape."""
    prefixes = ("color", "combined", "depth")

    prefixed_files = [fp for fp in folder.iterdir() if any(fp.name.startswith(p) for p in prefixes)]
    if not prefixed_files:
        raise RuntimeError(f"Didn't find any images in folder {folder} with prefixes {prefixes}!")

    image_groups: dict[ImageData, list[Path]] = {}
    for filepath in prefixed_files:
        filename = filepath.name
        for p in prefixes:
            if str(filename).startswith(p):
                remainder = filename[len(p) + 1 :]
                parts = remainder.split("==-")
                if len(parts) != 2:
                    print(f"Unexpected file name format: {filename}")
                    continue

                image_data = ImageData(waypoint_id=parts[0], image_type=p)
                image_groups.setdefault(image_data, []).append(filepath)
                break

    for data, paths in image_groups.items():
        print(f"Images of type {data.image_type} for waypoint {data.waypoint_id}:")
        for path in paths:
            try:
                img = Image.open(path)
                arr = np.array(img)
                print(f"    {path.name} -> dtype: {arr.dtype}, shape: {arr.shape}")
            except Exception as exc:
                print(f"    Error reading {path.name} as NumPy array: {exc}")


def introspect_pcd(filepath: Path) -> None:
    """Introspect the pointcloud in the given file."""
    assert filepath.suffix == ".pcd", f"Expected {filepath} to have the file suffix '.pcd'"
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])


def main() -> None:
    """Introspect the data in the given folder path."""
    parser = argparse.ArgumentParser(
        description="Introspect RGB-D dataset (pickle files and images).",
    )
    parser.add_argument("folder", type=Path, help="Path to the folder containing image files")
    args = parser.parse_args()
    folder: Path = args.folder
    assert folder.exists(), f"Folder {folder} does not exist."

    for filepath in folder.iterdir():
        if filepath.suffix == ".pkl":
            introspect_pickle(filepath)
            print("\n\n")

        if filepath.suffix == ".pcd":
            introspect_pcd(filepath)
            print("\n\n")

    introspect_images(folder)


if __name__ == "__main__":
    main()
