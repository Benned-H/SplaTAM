"""Refactor the SplaTAM codebase into modular functions."""

import numpy as np
import torch


def get_pointcloud(
    color,
    depth,
    intrinsics,
    w2c,
    transform_pts: bool = True,
    mask=None,
    compute_mean_sq_dist: bool = False,
    mean_sq_dist_method="projective",
):
    """Convert the given RGB-D image pair into a pointcloud.

    Originally from scripts/splatam.py.

    :param color: RGB image used to colorize the point cloud
    :param depth: Depth image used to construct the point cloud
    :param intrinsics: Matrix of camera intrinsic parameters (probably shape (3, 3))
    :param w2c: TODO
    :param transform_pts: TODO, defaults to True
    :param mask: TODO, defaults to None
    :param compute_mean_sq_dist: TODO, defaults to False
    :param mean_sq_dist_method: TODO, defaults to "projective"
    :raises ValueError: TODO
    :return: TODO
    """
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width).cuda().float(),
        torch.arange(height).cuda().float(),
        indexing="xy",
    )
    xx = (x_grid - CX) / FX
    yy = (y_grid - CY) / FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY) / 2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")

    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)  # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    return point_cld
