#!/usr/bin/env python3
"""Module: robot_data_loader

Provides utilities to load and process robot-collected RGB-D data,
construct an observation graph, optionally compress it, and
generate or cache a scene point cloud.
"""

import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import open3d as o3d
import torch
from PIL import Image


# function to compute rotation matrix from quaternion
def rotation_matrix_from_quaternion(quaternion):
    # Step 1: Normalize the quaternion
    quaternion = quaternion / np.linalg.norm(quaternion)

    # Step 2: Extract quaternion components
    w, x, y, z = quaternion

    # Step 3: Construct rotation matrix
    R = np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2],
        ],
    )
    return R


# Define the function for creating the point cloud
def get_pointcloud_from_graph_robot(
    graph: nx.Graph,
    chunk_size: int = 16,
    threshold: float = 0.9,
    downsample=False,
) -> o3d.geometry.PointCloud:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    nodes = list(graph.nodes(data=True))
    num_nodes = len(nodes)
    total_pcds = []
    total_colors = []

    for chunk_start in tqdm(range(0, num_nodes, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, num_nodes)
        chunk_nodes = nodes[chunk_start:chunk_end]

        rgbs, depths, poses = [], [], []

        for _, node_data in chunk_nodes:
            for i in range(4):
                rgb = node_data["rgb_tensor"][i]
                depth = torch.from_numpy(node_data["depth_data"][i]).float()
                pose = torch.from_numpy(node_data["pose_matrix"][i]).float()
                rgbs.append(rgb)
                depths.append(depth)
                poses.append(pose)

        print("datapoint count in chunk: ", len(rgbs))
        rgbs = torch.stack(rgbs).permute(0, 2, 3, 1).to(device)  # (B, H, W, 3)
        depths = torch.stack(depths).to(device)  # (B, 1, H, W)
        poses = torch.stack(poses).to(device)  # (B, 4, 4)

        # Use the previously defined function to get 3D coordinates
        xyz = spot_pixel_to_world_frame_batched(depths, poses)

        # Mask out low-confidence depth points and apply random sampling
        mask = (depths > 0) & (torch.rand(depths.shape, device=device) > threshold)
        rgbs, xyz = rgbs[mask.squeeze(1)], xyz[mask.squeeze(1)]

        total_colors.append(rgbs.cpu())
        total_pcds.append(xyz.cpu())

    total_pcds = torch.vstack(total_pcds)
    total_colors = torch.vstack(total_colors)

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(total_pcds.numpy())
    pcd_o3d.colors = o3d.utility.Vector3dVector(total_colors.numpy())

    if downsample:
        vl_size = 0.02
        print(f"Downsampling pointcloud || voxel_size: {vl_size}... ")
        pcd_o3d = pcd_o3d.voxel_down_sample(voxel_size=vl_size)
    else:
        print("Skipped downsampling ... ")

    return pcd_o3d


def compress_observation_graph(
    percentage_to_keep,
    observations_graph,
    group_by="direction",
    visualize=True,
    tmp_fldr="plots",
    plot_initial_graph=False,
):
    # Extract nodes from the graph
    nodes = [data for _, data in observations_graph.nodes(data=True)]

    if group_by == "direction":
        groups = group_nodes_by_direction(nodes, threshold_angle=10)
        selected_nodes = select_nodes_from_groups(groups, nodes, percentage_to_keep)
        if plot_initial_graph:
            plot_direction_groups(groups, visualize, tmp_fldr)  # Plot direction groups
        plot_nodes_side_by_side_direction(nodes, selected_nodes, groups, visualize, tmp_fldr)
    elif group_by == "position":
        node_coords, kmeans, y_kmeans, centers = cluster_nodes_by_position(
            nodes,
            percentage_to_keep,
        )
        groups = create_groups_from_clusters(nodes, y_kmeans, kmeans)
        selected_nodes = select_nodes_from_groups(groups, nodes, percentage_to_keep)
        if plot_initial_graph:
            plot_clustering(
                node_coords,
                y_kmeans,
                centers,
                [node["waypoint_key"] for node in nodes],
                visualize,
                tmp_fldr,
            )
        plot_nodes_side_by_side_position(
            nodes,
            selected_nodes,
            y_kmeans,
            centers,
            visualize,
            tmp_fldr,
        )
    elif group_by == "position_direction":
        node_coords, kmeans, y_kmeans, centers = cluster_nodes_by_position(
            nodes,
            percentage_to_keep,
        )
        groups = create_groups_from_clusters(nodes, y_kmeans, kmeans)
        selected_nodes = select_nodes_from_clusters_with_direction(
            groups,
            nodes,
            percentage_to_keep,
        )
        if plot_initial_graph:
            plot_clustering(
                node_coords,
                y_kmeans,
                centers,
                [node["waypoint_key"] for node in nodes],
                visualize,
                tmp_fldr,
            )
        direction_groups = [
            group_nodes_by_direction(cluster, threshold_angle=10) for cluster in groups
        ]
        flat_direction_groups = [
            group for subgroup in direction_groups for group in subgroup
        ]  # Flatten the nested list
        plot_nodes_side_by_side_position_direction(
            nodes,
            selected_nodes,
            y_kmeans,
            centers,
            flat_direction_groups,
            visualize,
            tmp_fldr,
        )

    # Create a new graph with the selected nodes
    new_observations_graph = nx.Graph()
    for node in selected_nodes:
        new_observations_graph.add_node(node["waypoint_key"], **node)

    return new_observations_graph


def load_robot_data(
    data_path: str,
    tmp_folder: str,
    pcd_downsample: bool = False,
    compression_percentage: float | None = None,
    compression_technique: str | None = None,
) -> tuple[
    o3d.geometry.PointCloud,
    nx.Graph,
    dict[int, str],
    dict[str, int],
    dict[int, tuple[float, float]],
]:
    """Load and process robot RGB-D data and poses into an observation graph.

    :param data_path: Path containing 'pose_data.pkl', 'pose_all_data.pkl', and RGB-D files
    :param tmp_folder: Folder to cache point cloud and waypoint outputs
    :param pcd_downsample: Whether to downsample the generated point cloud
    :param compression_percentage: Percent of nodes to drop (0-100)
    :param compression_technique: One of 'direction', 'position', 'position_direction'
    :returns: (environment point cloud, observation graph,
               node_id2key map, node_key2id map, node_coords map)
    """
    data_dir = Path(data_path)
    tmp_dir = Path(tmp_folder)

    # === Load pose dictionaries ===
    poses = _load_pickle(data_dir / "pose_data.pkl")
    all_poses = _load_pickle(data_dir / "pose_all_data.pkl")

    # === Initialize observation graph and mappings ===
    graph = nx.Graph()
    node_id2key: dict[int, str] = {}
    node_key2id: dict[str, int] = {}
    node_coords: dict[int, tuple[float, float]] = {}

    # === Populate nodes ===
    for node_id, key in enumerate(poses):
        node_id2key[node_id] = key
        node_key2id[key] = node_id

        rgb_imgs, rgb_tensors, depths, pose_dicts, pose_mats = _load_waypoint_data(
            data_dir,
            key,
            all_poses,
        )

        # Representative pose entry
        rep_pose = poses[key]
        rep_pose["rotation_matrix"] = rotation_matrix_from_quaternion(
            rep_pose["quaternion(wxyz)"],
        )
        xy = tuple(rep_pose["position"][:2])

        # Add node with all data attached
        graph.add_node(
            node_id,
            rgb=rgb_imgs,
            rgb_tensor=rgb_tensors,
            depth_data=depths,
            pose=pose_dicts,
            pose_matrix=pose_mats,
            xy_coordinate=xy,
            waypoint_key=key,
            rep_pose=rep_pose,
        )
        node_coords[node_id] = xy

    # === Optional compression ===
    if compression_percentage is not None:
        _compress_graph(
            graph,
            compression_percentage,
            compression_technique,
            tmp_dir,
        )

    # === Point cloud generation or loading ===
    env_pcd = _get_or_create_pointcloud(
        graph,
        tmp_dir,
        downsample=pcd_downsample,
    )

    # === Save waypoints mapping ===
    tmp_dir.mkdir(parents=True, exist_ok=True)
    np.save(tmp_dir / "waypoints.npy", dict(graph.nodes(data="pose")))

    return env_pcd, graph, node_id2key, node_key2id, node_coords


# ------------------------------------------------------------------


def _load_pickle(path: Path) -> dict:
    """Load a pickle file and return its contents.

    :param path: Path to .pkl file
    :raises FileNotFoundError: if file missing
    :returns: Unpickled object
    """
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_waypoint_data(
    data_dir: Path,
    waypoint: str,
    all_poses: dict,
) -> tuple[
    dict[int, Image.Image],
    dict[int, torch.Tensor],
    dict[int, np.ndarray],
    dict[int, dict],
    dict[int, np.ndarray],
]:
    """Load color images, tensors, depth arrays, and pose matrices for 4 orientations.

    :param data_dir: Base folder for data
    :param waypoint: Waypoint identifier
    :param all_poses: Full pose dictionary with directional keys
    :returns: (rgb images, rgb tensors, depths, raw poses, pose matrices)
    """
    rgb_imgs, rgb_tensors, depths, poses, mats = {}, {}, {}, {}, {}

    for ori in range(4):
        # --- RGB ---
        img_file = data_dir / f"color_{waypoint}-{ori}.jpg"
        img = Image.open(img_file).convert("RGB")
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        # --- Depth ---
        depth_file = data_dir / f"depth_{waypoint}-{ori}.npy"
        depth_arr = np.load(depth_file, allow_pickle=True)

        # --- Pose and matrix ---
        key = f"{waypoint}-{ori}"
        pose_dict = all_poses[key]
        R = rotation_matrix_from_quaternion(pose_dict["quaternion(wxyz)"])
        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3] = pose_dict["position"]
        pose_dict["rotation_matrix"] = R

        rgb_imgs[ori] = img
        rgb_tensors[ori] = tensor
        depths[ori] = depth_arr
        poses[ori] = pose_dict
        mats[ori] = mat

    return rgb_imgs, rgb_tensors, depths, poses, mats


def _compress_graph(
    graph: nx.Graph,
    percentage: float,
    technique: str | None,
    tmp_dir: Path,
) -> None:
    """Compress the observation graph by dropping nodes.

    :param graph: The observation graph
    :param percentage: Percent of nodes to drop
    :param technique: One of ['direction','position','position_direction']
    :param tmp_dir: Folder for temporary files
    :raises ValueError: on invalid technique
    """
    valid = ["direction", "position", "position_direction"]
    if technique not in valid:
        raise ValueError(f"Invalid compression technique: {technique}, choose from {valid}.")
    keep = 100 - percentage
    before = len(graph.nodes)
    compressed = compress_observation_graph(
        keep,
        graph,
        group_by=technique,
        visualize=False,
        tmp_fldr=str(tmp_dir),
    )
    graph.clear()
    graph.update(compressed)
    after = len(graph.nodes)
    print(f"Compressed graph ({technique}): {before} -> {after} nodes")


def _get_or_create_pointcloud(
    graph: nx.Graph,
    tmp_dir: Path,
    downsample: bool = False,
) -> o3d.geometry.PointCloud:
    """Load or generate a point cloud from the observation graph.

    :param graph: The observation graph
    :param tmp_dir: Folder for caching
    :param downsample: Whether to downsample the point cloud
    :returns: Open3D PointCloud object
    """
    pcd_file = tmp_dir / "pointcloud.pcd"
    if pcd_file.exists():
        print(f"Loading point cloud: {pcd_file}")
        return o3d.io.read_point_cloud(str(pcd_file))

    print("Generating new point cloud from observations...")
    pcd = get_pointcloud_from_graph_robot(graph, downsample)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(pcd_file), pcd)
    print(f"Saved point cloud to {pcd_file}")
    return pcd
