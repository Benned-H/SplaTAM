# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Command line interface for graph nav with options to download/upload a map and to navigate a map."""

import argparse
import logging
import math
import os
import pickle
import sys
import time

import bosdyn.client.channel
import bosdyn.client.util
import cv2
import google.protobuf.timestamp_pb2
import grpc
import numpy as np
from bosdyn.api import geometry_pb2, power_pb2, robot_state_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.client import ResponseError, RpcError, math_helpers
from bosdyn.client.exceptions import ResponseError
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
    get_frame_names,
    get_odom_tform_body,
    get_se2_a_tform_b,
)
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.power import PowerClient, power_on, safe_power_off
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    block_until_arm_arrives,
    blocking_stand,
)
from bosdyn.client.robot_state import RobotStateClient
from tqdm import tqdm

from utils import graph_nav_util
from utils.tsp import draw_observations_graph, shortest_waypoint_path


class GraphNavInterface:
    """GraphNav service command line interface."""

    def __init__(self, robot, upload_path, data_directory_path):
        self._robot = robot

        self._image_client = self._robot.ensure_client(ImageClient.default_service_name)
        self._sources = ["hand_depth_in_hand_color_frame", "hand_color_image"]
        self._data_directory_path = data_directory_path

        # Force trigger timesync.
        self._robot.time_sync.wait_for_sync()

        # Create robot state and command clients.
        self._robot_command_client = self._robot.ensure_client(
            RobotCommandClient.default_service_name,
        )
        self._robot_state_client = self._robot.ensure_client(RobotStateClient.default_service_name)

        # Create the client for the Graph Nav main service.
        self._graph_nav_client = self._robot.ensure_client(GraphNavClient.default_service_name)

        # Create a power client for the robot.
        self._power_client = self._robot.ensure_client(PowerClient.default_service_name)

        # Boolean indicating the robot's power state.
        power_state = self._robot_state_client.get_robot_state().power_state
        self._started_powered_on = power_state.motor_power_state == power_state.STATE_ON
        self._powered_on = self._started_powered_on

        # Number of attempts to wait before trying to re-power on.
        self._max_attempts_to_wait = 50

        # Store the most recent knowledge of the state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = dict()  # maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = dict()

        # Filepath for uploading a saved graph's and snapshots too.
        if upload_path[-1] == "/":
            self._upload_filepath = upload_path[:-1]
        else:
            self._upload_filepath = upload_path

        self._command_dictionary = {
            "1": self._get_localization_state,
            "2": self._set_initial_localization_fiducial,
            "3": self._set_initial_localization_waypoint,
            "4": self._list_graph_waypoint_and_edge_ids,
            "5": self._upload_graph_and_snapshots,
            "6": self._navigate_to,
            "7": self._navigate_route,
            "8": self._navigate_to_anchor,
            "9": self._clear_graph,
            "10": self._collect_data_all_waypoints,
            "11": self._create_edge_datastructure,
            "12": self._get_anchor_tform_waypoints,
            "13": self._download_full_graph,
            "14": self._compress_graph,
            "15": self._add_compress_graph,
        }

    def _get_transform(self, from_wp, to_wp):
        """Get transform from from-waypoint to to-waypoint."""
        from_se3 = from_wp.waypoint_tform_ko
        from_tf = SE3Pose(
            from_se3.position.x,
            from_se3.position.y,
            from_se3.position.z,
            Quat(
                w=from_se3.rotation.w,
                x=from_se3.rotation.x,
                y=from_se3.rotation.y,
                z=from_se3.rotation.z,
            ),
        )

        to_se3 = to_wp.waypoint_tform_ko
        to_tf = SE3Pose(
            to_se3.position.x,
            to_se3.position.y,
            to_se3.position.z,
            Quat(
                w=to_se3.rotation.w,
                x=to_se3.rotation.x,
                y=to_se3.rotation.y,
                z=to_se3.rotation.z,
            ),
        )

        from_T_to = from_tf.mult(to_tf.inverse())
        return from_T_to.to_proto()

    def _get_waypoint(self, id):
        """Get waypoint from graph (return None if waypoint not found)"""
        if self._current_graph is None:
            self._current_graph = self._graph_nav_client.download_graph()

        for waypoint in self._current_graph.waypoints:
            if waypoint.id == id:
                return waypoint

        print(f"ERROR: Waypoint {id} not found in graph.")
        return None

    def _update_graph_waypoint_and_edge_ids(self, do_print=False):
        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print("Empty graph.")
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = (
            graph_nav_util.update_waypoints_and_edges(graph, localization_id, do_print)
        )

    def _open_gripper_and_stow(self):
        gripper_open = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
        cmd_id = self._robot_command_client.robot_command(gripper_open)
        self.stow_arm()

    def stow_arm(self):
        # Stow the arm
        # Build the stow command using RobotCommandBuilder
        stow = RobotCommandBuilder.arm_stow_command()

        # Issue the command via the RobotCommandClient
        stow_command_id = self._robot_command_client.robot_command(stow)

        block_until_arm_arrives(self._robot_command_client, stow_command_id, 3.0)

    def _create_new_edge(self, *args):
        """Create new edge between existing waypoints in map."""
        if len(args[0]) != 2:
            print("ERROR: Specify the two waypoints to connect (short code or annotation).")
            return

        self._update_graph_waypoint_and_edge_ids(do_print=False)

        from_id = graph_nav_util.find_unique_waypoint_id(
            args[0][0],
            self._current_graph,
            self._current_annotation_name_to_wp_id,
        )
        to_id = graph_nav_util.find_unique_waypoint_id(
            args[0][1],
            self._current_graph,
            self._current_annotation_name_to_wp_id,
        )

        print(f"Creating edge from {from_id} to {to_id}.")

        from_wp = self._get_waypoint(from_id)
        if from_wp is None:
            return

        to_wp = self._get_waypoint(to_id)
        if to_wp is None:
            return

        # Get edge transform based on kinematic odometry
        edge_transform = self._get_transform(from_wp, to_wp)

        # Define new edge
        new_edge = map_pb2.Edge()
        new_edge.id.from_waypoint = from_id
        new_edge.id.to_waypoint = to_id
        new_edge.from_tform_to.CopyFrom(edge_transform)

        print("edge transform =", new_edge.from_tform_to)

        # Send request to add edge to map
        self._recording_client.create_edge(edge=new_edge)

    def _add_compress_graph(self, *args):
        for (
            representative_name,
            connected_representatives_list,
        ) in self._compressed_edge_connectivity.items():
            for connected_representative in connected_representatives_list:
                self._create_new_edge([representative_name, connected_representative])

    def _compress_graph(self, *args):
        # import sys
        # sys.path.insert(0,'/home/eric/Github/VLGMP')

        # Import vlgmp library
        # from vlgmp.perception.vlm_library import vlm_library

        import pickle

        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np
        from PIL import Image

        observation_data = {"images": {}, "poses": {}}

        observations = []
        # load pose data
        with open(f"{self._data_directory_path}/pose_data.pkl", "rb") as f:
            poses = pickle.load(f)

        # get waypoint edge connectivity
        with open(f"{self._data_directory_path}/connectivty_cost_dict.pkl", "rb") as f:
            edge_connectivity = pickle.load(f)

        # get corresponding images for each pose
        for id, waypoint_name in enumerate(poses.keys()):
            print(f"{id} out of {len(poses.keys())} || Getting image for waypoint:{waypoint_name}")
            ###image = Image.open(f'{data_path}/color_{waypoint_name}.jpg').convert("RGB")
            observation_data["images"][waypoint_name] = None  ##image
            observation_data["poses"][waypoint_name] = poses[waypoint_name]

        # Code to create a graph
        observations_graph = nx.Graph()
        node_id2key = {}
        node_key2id = {}
        node_coords = {}

        for i, node_key in enumerate(observation_data["images"].keys()):
            node_id2key[i] = node_key
            node_key2id[node_key] = i

            ###print(f"Adding node {i} with key {node_key} to graph")

            node_image = observation_data["images"][node_key]
            node_pose = observation_data["poses"][node_key]
            coord = tuple(node_pose["position"][0:2])  # x,y axis from position
            # coord = tuple(node_pose['quaternion(wxyz)'][1:3]) #x,y axis from quaternion

            observations_graph.add_node(
                node_for_adding=i,
                rgb=node_image,
                pose=node_pose,
                xy_coordinate=coord,
            )
            node_coords[i] = coord

        ##  Manually setting edges all noded connected to everyother node|| weight of edges from each node to every other node is ecludian distance between the nodes
        # for node_idx, xy in enumerate(node_coords):
        #     for other_node_idx, other_xy in enumerate(node_coords):
        #         if node_idx != other_node_idx:
        #             ecludiean_distance = np.linalg.norm(np.array(xy) - np.array(other_xy))
        #             rounded = round(ecludiean_distance, 2)
        #             observations_graph.add_edge(node_idx, other_node_idx, distance=rounded)

        # weights of edges from waypoints edge connectivity data

        for waypoint_name in edge_connectivity:
            for connected_waypoint in edge_connectivity[waypoint_name]:
                # print(f"waypoint is: {waypoint_name} || connected waypoint is: {connected_waypoint}")

                origin_node_id, connected_node_id = (
                    node_key2id[waypoint_name],
                    node_key2id[connected_waypoint[0]],
                )
                origin_node_xy, connected_node_xy = (
                    node_coords[origin_node_id],
                    node_coords[connected_node_id],
                )

                ecludiean_distance = np.linalg.norm(
                    np.array(origin_node_xy) - np.array(connected_node_xy),
                )
                rounded = round(ecludiean_distance, 2)
                # print(f"Waypoint id: {str(origin_node_id):2s} || name: {waypoint_name:40s} connected to: Waypoint id: {str(connected_node_id):2s} || name: {connected_waypoint[0]:40s} with cost: {ecludiean_distance}")
                # print(f"Waypoint id: {str(origin_node_id):2s} || name: {waypoint_name:40s} connected to: Waypoint id: {str(connected_node_id):2s} || name: {connected_waypoint[0]:40s} with cost: {connected_waypoint[1]}")
                observations_graph.add_edge(origin_node_id, connected_node_id, distance=rounded)

        draw_observations_graph(observations_graph, node_coords, plt_size=(10, 10), axis=False)

        # clustering to extract spacial representative waypoints while reducing overall number of waypoints
        # import seaborn as sns; sns.set()  # for plot styling
        import numpy as np
        import pandas as pd
        from scipy.spatial.distance import cdist
        from sklearn.cluster import KMeans
        from sklearn.manifold import TSNE

        node_id2key = {}
        node_key2id = {}
        node_coords = {}
        node_poses = {}
        node_images = {}

        for i, node_key in enumerate(observation_data["images"].keys()):
            node_id2key[i] = node_key
            node_key2id[node_key] = i

            print(f"Node {i} with key {node_key}")

            node_image = observation_data["images"][node_key]
            node_pose = observation_data["poses"][node_key]
            coord = tuple(node_pose["position"][0:2])  # x,y axis from position
            # coord = tuple(node_pose['quaternion(wxyz)'][1:3]) #x,y axis from quaternion
            node_coords[i] = coord
            node_poses[i] = node_pose
            node_images[i] = node_image

        # original scatter plot of nodes based on coordinates
        x = [coord[0] for coord in node_coords.values()]
        y = [coord[1] for coord in node_coords.values()]
        labels = [str(key) for key in node_coords.keys()]

        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the width and height as desired

        ax.scatter(x, y)
        for label, x_coord, y_coord in zip(labels, x, y, strict=False):
            ax.annotate(
                label,
                (x_coord, y_coord),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_title("Scatter Plot")

        plt.show()

        # clustering
        list_of_embeddings = list(node_coords.values())

        num_centroids = 8

        # kmeans = KMeans(n_clusters = exploration_zones, init='k-means++', random_state=42)
        kmeans = KMeans(n_clusters=num_centroids, init="k-means++", random_state=1)
        kmeans.fit(list_of_embeddings)
        y_kmeans = kmeans.predict(list_of_embeddings)

        # store clusters in dataframe
        cluster_map = pd.DataFrame()
        cluster_map["cluster_index"] = y_kmeans
        cluster_map["node_id"] = labels

        x = [coord[0] for coord in node_coords.values()]
        y = [coord[1] for coord in node_coords.values()]
        labels = [str(key) for key in node_coords.keys()]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x, y, c=y_kmeans, s=50, cmap="viridis")  # plot embeddingpoints

        centers = kmeans.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5)  # plot centers

        # plt.scatter()
        plt.title("Clustering waypoints based on coordinates")

        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i], y[i]))

        # iterate over each centroid and find the closest datapoint within its cluster. We can also account for selecting the waypoint with the most edges
        representative_waypoints = {}
        representative_waypoints_names = []
        closest_labels = []
        for i, centroid in enumerate(centers):
            idx = np.where(y_kmeans == i)[0]  # Get the indices of points in the cluster
            cluster_points = [list_of_embeddings[j] for j in idx]
            cluster_labels = [labels[j] for j in idx]
            distances = cdist([centroid], cluster_points, metric="euclidean")[0]
            closest_datapoint = cluster_points[np.argmin(distances)]
            closest_label = cluster_labels[np.argmin(distances)]
            representative_waypoints[int(closest_label)] = closest_datapoint

            representative_waypoints_names.append(node_id2key[int(closest_label)])

        # iterate over centroids and select node from class with most edges from edgeconnectivity
        # for i, centroid in enumerate(centers):
        #     idx = np.where(y_kmeans == i)[0]  # Get the indices of points in the cluster
        #     cluster_points = [list_of_embeddings[j] for j in idx]
        #     cluster_labels = [labels[j] for j in idx]
        #     num_edges=[ len(edge_connectivity[node_id2key[int(label)]]) for label in cluster_labels]

        #     most_connected_datapoint = cluster_points[np.argmax(num_edges)]
        #     most_connected_dlabel = cluster_labels[np.argmax(num_edges)]
        #     representative_waypoints[int(most_connected_dlabel)] = most_connected_datapoint

        print("Spacial representative waypoints: ")
        representative_waypoints

        # print(representative_waypoints_names)
        waypoint_to_representative_names = {}

        # first, I need a dictionary that maps represenative waypoints, to to the list of waypoints in it's cluster
        representative_to_waypoints_names = {}  # key will be id of (representative) waypoint, value will be associated waypoints in cluster
        representative_to_waypoints_ids = {}  # key will be id of (representative) waypoint, value will be list of connected (represenative) waypoints
        for (
            representative_waypoint_id
        ) in representative_waypoints:  # for each representative waypoint
            representative_to_waypoints_names[node_id2key[representative_waypoint_id]] = []
            representative_to_waypoints_ids[representative_waypoint_id] = []
            # find all waypoints that are represented by this waypoint
            for waypoint_id, waypoint_cluster_num in enumerate(
                y_kmeans,
            ):  # waypoint_id is number id of waypoint, waypoint_cluster_num is the cluster number
                associated_represenative_waypoint_name = representative_waypoints_names[
                    waypoint_cluster_num
                ]
                if (
                    associated_represenative_waypoint_name
                    == node_id2key[representative_waypoint_id]
                ):
                    # print(f"{waypoint_id} is associated with {representative_waypoint_id}")
                    representative_to_waypoints_names[
                        node_id2key[representative_waypoint_id]
                    ].append(node_id2key[waypoint_id])
                    representative_to_waypoints_ids[representative_waypoint_id].append(waypoint_id)

                    waypoint_to_representative_names[node_id2key[waypoint_id]] = node_id2key[
                        representative_waypoint_id
                    ]

        # print(representative_to_waypoints_ids)

        new_connectivity_graph_names = {}  # keys are (representative) waypoints, value is list of waypoints it should be connected to since some waypoint in it's cluster was connected to it
        new_connectivity_graph_ids = {}
        for (
            representative_waypoint_name,
            list_of_represented_waypoints,
        ) in representative_to_waypoints_names.items():
            new_connectivity_graph_names[representative_waypoint_name] = []
            new_connectivity_graph_ids[node_key2id[representative_waypoint_name]] = []
            # print(f"reprensetative waypoint: {representative_waypoint_name} | represented waypoints: {list_of_represented_waypoints}")
            for (
                represented_waypoint_name
            ) in list_of_represented_waypoints:  # look at each waypoint represented in this node
                if (
                    represented_waypoint_name in edge_connectivity
                ):  # only do this if the represented waypoint is connected to something, TODO: why is that the case?
                    for connected_waypoint, cost in edge_connectivity[
                        represented_waypoint_name
                    ]:  # for every waypoint the representative is connected to
                        # print(f"represented waypoint: {represented_waypoint_name} - connected waypoint: {connected_waypoint}")
                        # print(f"represented waypoint (id) {node_key2id[represented_waypoint_name]} - connected waypoint(id) {node_key2id[connected_waypoint]}")
                        # let's find the connected_waypoints representative, and make this represenative connected to it!
                        if (
                            representative_waypoint_name
                            != waypoint_to_representative_names[connected_waypoint]
                        ):  # no self loops
                            if (
                                waypoint_to_representative_names[connected_waypoint]
                                not in new_connectivity_graph_names[representative_waypoint_name]
                            ):  # no repeats
                                new_connectivity_graph_names[representative_waypoint_name].append(
                                    waypoint_to_representative_names[connected_waypoint],
                                )
                                new_connectivity_graph_ids[
                                    node_key2id[representative_waypoint_name]
                                ].append(
                                    node_key2id[
                                        waypoint_to_representative_names[connected_waypoint]
                                    ],
                                )

        print(f"new connectiviy graph_ids: {new_connectivity_graph_ids}")
        print(f"new connectiviy graph_names: {new_connectivity_graph_names}")

        self._compressed_edge_connectivity = new_connectivity_graph_names

        new_observations_graph = nx.Graph()

        for i in new_connectivity_graph_ids:
            node_key = node_id2key[i]

            print(f"Adding node {i} with key {node_key} to graph")

            node_image = observation_data["images"][node_key]
            node_pose = observation_data["poses"][node_key]
            coord = tuple(node_pose["position"][0:2])  # x,y axis from position
            # coord = tuple(node_pose['quaternion(wxyz)'][1:3]) #x,y axis from quaternion

            new_observations_graph.add_node(
                node_for_adding=i,
                rgb=node_image,
                pose=node_pose,
                xy_coordinate=coord,
            )
            node_coords[i] = coord

        ##  Manually setting edges all noded connected to everyother node|| weight of edges from each node to every other node is ecludian distance between the nodes
        # for node_idx, xy in enumerate(node_coords):
        #     for other_node_idx, other_xy in enumerate(node_coords):
        #         if node_idx != other_node_idx:
        #             ecludiean_distance = np.linalg.norm(np.array(xy) - np.array(other_xy))
        #             rounded = round(ecludiean_distance, 2)
        #             observations_graph.add_edge(node_idx, other_node_idx, distance=rounded)

        # weights of edges from waypoints edge connectivity data

        for waypoint_id in new_connectivity_graph_ids:
            for connected_waypoint_id in new_connectivity_graph_ids[waypoint_id]:
                # print(f"waypoint is: {waypoint_name} || connected waypoint is: {connected_waypoint}")

                origin_node_id, connected_node_id = waypoint_id, connected_waypoint_id
                origin_node_xy, connected_node_xy = (
                    node_coords[origin_node_id],
                    node_coords[connected_node_id],
                )

                ecludiean_distance = np.linalg.norm(
                    np.array(origin_node_xy) - np.array(connected_node_xy),
                )
                rounded = round(ecludiean_distance, 2)
                # print(f"Waypoint id: {str(origin_node_id):2s} || name: {waypoint_name:40s} connected to: Waypoint id: {str(connected_node_id):2s} || name: {connected_waypoint[0]:40s} with cost: {ecludiean_distance}")
                # print(f"Waypoint id: {str(origin_node_id):2s} || name: {waypoint_name:40s} connected to: Waypoint id: {str(connected_node_id):2s} || name: {connected_waypoint[0]:40s} with cost: {connected_waypoint[1]}")
                new_observations_graph.add_edge(origin_node_id, connected_node_id, distance=rounded)

        draw_observations_graph(new_observations_graph, node_coords, plt_size=(10, 10), axis=False)

    def _download_and_write_edge_snapshots(self, edges):
        """Download the edge snapshots from robot to the specified, local filepath location."""
        num_edge_snapshots_downloaded = 0
        num_to_download = 0
        for edge in edges:
            if len(edge.snapshot_id) == 0:
                continue
            num_to_download += 1
            try:
                edge_snapshot = self._graph_nav_client.download_edge_snapshot(edge.snapshot_id)
            except Exception:
                # Failure in downloading edge snapshot. Continue to next snapshot.
                print("Failed to download edge snapshot: " + edge.snapshot_id)
                continue
            self._write_bytes(
                self._download_filepath + "/edge_snapshots",
                "/" + edge.snapshot_id,
                edge_snapshot.SerializeToString(),
            )
            num_edge_snapshots_downloaded += 1
            print(
                f"Downloaded {num_edge_snapshots_downloaded} of the total {num_to_download} edge snapshots.",
            )

    def _download_and_write_waypoint_snapshots(self, waypoints):
        """Download the waypoint snapshots from robot to the specified, local filepath location."""
        num_waypoint_snapshots_downloaded = 0
        for waypoint in waypoints:
            if len(waypoint.snapshot_id) == 0:
                continue
            try:
                waypoint_snapshot = self._graph_nav_client.download_waypoint_snapshot(
                    waypoint.snapshot_id,
                )
            except Exception:
                # Failure in downloading waypoint snapshot. Continue to next snapshot.
                print("Failed to download waypoint snapshot: " + waypoint.snapshot_id)
                continue
            self._write_bytes(
                self._download_filepath + "/waypoint_snapshots",
                "/" + waypoint.snapshot_id,
                waypoint_snapshot.SerializeToString(),
            )
            num_waypoint_snapshots_downloaded += 1
            print(
                f"Downloaded {num_waypoint_snapshots_downloaded} of the total {len(waypoints)} waypoint snapshots.",
            )

    def _write_bytes(self, filepath, filename, data):
        """Write data to a file."""
        os.makedirs(filepath, exist_ok=True)
        with open(filepath + filename, "wb+") as f:
            f.write(data)
            f.close()

    def _write_full_graph(self, graph):
        """Download the graph from robot to the specified, local filepath location."""
        graph_bytes = graph.SerializeToString()
        self._write_bytes(self._download_filepath, "/graph", graph_bytes)

    def _download_full_graph(self, *args):
        """Download the graph and snapshots from the robot."""
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print("Failed to download the graph.")
            return
        self._write_full_graph(graph)
        print(
            f"Graph downloaded with {len(graph.waypoints)} waypoints and {len(graph.edges)} edges",
        )
        # Download the waypoint and edge snapshots.
        self._download_and_write_waypoint_snapshots(graph.waypoints)
        self._download_and_write_edge_snapshots(graph.edges)

    def _get_anchor_tform_waypoints(self, *args):
        pose_dict = {}
        for anchor in self._current_graph.anchoring.anchors:
            pos = anchor.seed_tform_waypoint.position
            rot = anchor.seed_tform_waypoint.rotation

            quat = bosdyn.client.math_helpers.Quat(w=rot.w, x=rot.x, y=rot.y, z=rot.z)
            yaw = quat.to_yaw()
            print(f"id: {anchor.id} x: {pos.x} y: {pos.y} z: {pos.z} yaw (radians): {yaw}")
            pose_dict[anchor.id] = {
                "position": [pos.x, pos.y, pos.z],
                "quaternion(wxyz)": [quat.w, quat.x, quat.y, quat.z],
            }
        pickle.dump(pose_dict, open(f"{self._data_directory_path}/pose_data.pkl", "wb"))

    def _create_edge_datastructure(self, *args):
        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id
        graph = self._graph_nav_client.download_graph()

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = (
            graph_nav_util.update_waypoints_and_edges(graph, localization_id)
        )

        edge_cost_dictionary = {}

        """
        for edge in graph.edges:
            edge_cost_dictionary[edge.id.from_waypoint] = [edge.id.to_waypoint, edge.annotations.cost.value]
        """

        # keys are waypoint names, values are tuples (connected_waypoint,cost)

        for waypoint_idx, waypoint_name in enumerate(self._current_edges):
            edge_cost_dictionary[waypoint_name] = []
            for connected_waypoint in self._current_edges[waypoint_name]:
                edge_cost_dictionary[waypoint_name].append([connected_waypoint, 0])

        print(edge_cost_dictionary)
        pickle.dump(
            edge_cost_dictionary,
            open(f"{self._data_directory_path}/connectivty_cost_dict.pkl", "wb"),
        )

    def _collect_data_all_waypoints(self, *args):
        # Power on the robot
        self.toggle_power(should_power_on=True)

        # stow the robot arm and open the gripper
        self._open_gripper_and_stow()

        # first, get the waypoints and edges
        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print("Empty graph.")
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = (
            graph_nav_util.update_waypoints_and_edges(graph, localization_id)
        )

        with open(f"{self._data_directory_path}/pose_data.pkl", "rb") as f:
            poses = pickle.load(f)

        pose_data_all = {}

        # for waypoint_truename in tqdm(list(poses.keys())):
        # Calculate shortest path to all waypoints and loop waypoint_truename over that
        for waypoint_truename in tqdm(shortest_waypoint_path(self._data_directory_path)):
            print(f"going to {waypoint_truename}")
            # navigate to the waypoint
            self._navigate_to_reuse([waypoint_truename])

            ## The robot will take 4 images at each waypoint, at 4 different angles (each 90 degrees apart)
            for i in range(4):
                print(f"Collecting data at angle {i}")
                # take sensor data
                image_responses = self._image_client.get_image_from_sources(self._sources)

                cv_depth = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint16)
                cv_depth = cv_depth.reshape(
                    image_responses[0].shot.image.rows,
                    image_responses[0].shot.image.cols,
                )

                # cv_depth is in millimeters, divide by 1000 to get it into meters
                cv_depth_meters = cv_depth / 1000.0

                # Visual is a JPEG
                cv_visual = cv2.imdecode(
                    np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8),
                    -1,
                )

                # cv2.imwrite("color.jpg", cv_visual)

                # Convert the visual image from a single channel to RGB so we can add color
                visual_rgb = (
                    cv_visual
                    if len(cv_visual.shape) == 3
                    else cv2.cvtColor(cv_visual, cv2.COLOR_GRAY2RGB)
                )
                # Map depth ranges to color

                # cv2.applyColorMap() only supports 8-bit; convert from 16-bit to 8-bit and do scaling
                min_val = np.min(cv_depth)
                max_val = np.max(cv_depth)
                depth_range = max_val - min_val
                depth8 = (255.0 / depth_range * (cv_depth - min_val)).astype("uint8")
                depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
                depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)

                # Add the two images together.
                out = cv2.addWeighted(visual_rgb, 0.5, depth_color, 0.5, 0)

                cv2.imwrite(
                    f"{self._data_directory_path}/color_{waypoint_truename}-{i}.jpg",
                    cv_visual,
                )
                pickle.dump(
                    cv_depth_meters,
                    open(f"{self._data_directory_path}/depth_{waypoint_truename}-{i}", "wb"),
                )
                cv2.imwrite(
                    f"{self._data_directory_path}/combined_{waypoint_truename}-{i}.jpg",
                    out,
                )

                # Get pose information
                state = self._graph_nav_client.get_localization_state()
                seed_tform_hand = state.localization.seed_tform_body

                # dump pose data
                pose_data_all[f"{waypoint_truename}-{i}"] = {
                    "position": [
                        seed_tform_hand.position.x,
                        seed_tform_hand.position.y,
                        seed_tform_hand.position.z,
                    ],
                    "quaternion(wxyz)": [
                        seed_tform_hand.rotation.w,
                        seed_tform_hand.rotation.x,
                        seed_tform_hand.rotation.y,
                        seed_tform_hand.rotation.z,
                    ],
                }

                pickle.dump(
                    pose_data_all,
                    open(f"{self._data_directory_path}/pose_all_data.pkl", "wb"),
                )

                # turn robot 90 degrees
                if i != 3:  # don't turn on the last iteration
                    self._turn_left()

    def _turn_left(self):
        # print("inside turn left")
        frame_name = ODOM_FRAME_NAME
        stairs = False
        transforms = self._robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
        # print(f"transforms: {transforms}")

        # Build the transform for where we want the robot to be relative to where the body currently is.
        body_tform_goal = math_helpers.SE2Pose(x=0, y=0, angle=1.57)
        # We do not want to command this goal in body frame because the body will move, thus shifting
        # our goal. Instead, we transform this offset to get the goal position in the output frame
        # (which will be either odom or vision).
        out_tform_body = get_se2_a_tform_b(transforms, frame_name, BODY_FRAME_NAME)
        print(f"out_tform_body: {out_tform_body}, body_tform_goal: {body_tform_goal}")
        out_tform_goal = out_tform_body * body_tform_goal

        print("made it here 1")

        # Command the robot to go to the goal point in the specified frame. The command will stop at the
        # new position.
        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=out_tform_goal.x,
            goal_y=out_tform_goal.y,
            goal_heading=out_tform_goal.angle,
            frame_name=frame_name,
            params=RobotCommandBuilder.mobility_params(stair_hint=stairs),
        )
        end_time = 10.0
        cmd_id = self._robot_command_client.robot_command(
            lease=None,
            command=robot_cmd,
            end_time_secs=time.time() + end_time,
        )
        # Wait until the robot has reached the goal.
        while True:
            feedback = self._robot_command_client.robot_command_feedback(cmd_id)
            mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
            if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
                print("Failed to reach the goal")
                return False
            traj_feedback = mobility_feedback.se2_trajectory_feedback
            if (
                traj_feedback.status == traj_feedback.STATUS_AT_GOAL
                and traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED
            ):
                print("Arrived at the goal.")
                return True
            time.sleep(1)

    def _get_localization_state(self, *args):
        """Get the current localization and state of the robot."""
        state = self._graph_nav_client.get_localization_state()
        print("Got localization: \n%s" % str(state.localization))
        odom_tform_body = get_odom_tform_body(state.robot_kinematics.transforms_snapshot)
        print("Got robot state in kinematic odometry frame: \n%s" % str(odom_tform_body))

    def _set_initial_localization_fiducial(self, *args):
        """Trigger localization when near a fiducial."""
        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot,
        ).to_proto()
        # Create an empty instance for initial localization since we are asking it to localize
        # based on the nearest fiducial.
        localization = nav_pb2.Localization()
        self._graph_nav_client.set_localization(
            initial_guess_localization=localization,
            ko_tform_body=current_odom_tform_body,
        )

    def _set_initial_localization_waypoint(self, *args):
        """Trigger localization to a waypoint."""
        # Take the first argument as the localization waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without initializing.
            print("No waypoint specified to initialize to.")
            return
        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0],
            self._current_graph,
            self._current_annotation_name_to_wp_id,
        )
        if not destination_waypoint:
            # Failed to find the unique waypoint id.
            return

        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot,
        ).to_proto()
        # Create an initial localization to the specified waypoint as the identity.
        localization = nav_pb2.Localization()
        localization.waypoint_id = destination_waypoint
        localization.waypoint_tform_body.rotation.w = 1.0
        self._graph_nav_client.set_localization(
            initial_guess_localization=localization,
            # It's hard to get the pose perfect, search +/-20 deg and +/-20cm (0.2m).
            max_distance=0.2,
            max_yaw=20.0 * math.pi / 180.0,
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
            ko_tform_body=current_odom_tform_body,
        )

    def _list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the robot."""
        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print("Empty graph.")
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = (
            graph_nav_util.update_waypoints_and_edges(graph, localization_id)
        )

    def _upload_graph_and_snapshots(self, *args):
        """Upload the graph and snapshots to the robot."""
        print("Loading the graph from disk into local storage...")
        with open(self._upload_filepath + "/graph", "rb") as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)
            print(
                f"Loaded graph has {len(self._current_graph.waypoints)} waypoints and {len(self._current_graph.edges)} edges",
            )
        for waypoint in self._current_graph.waypoints:
            # Load the waypoint snapshots from disk.
            with open(
                self._upload_filepath + f"/waypoint_snapshots/{waypoint.snapshot_id}",
                "rb",
            ) as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                self._current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
        for edge in self._current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            # Load the edge snapshots from disk.
            with open(
                self._upload_filepath + f"/edge_snapshots/{edge.snapshot_id}",
                "rb",
            ) as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        print("Uploading the graph and snapshots to the robot...")
        true_if_empty = not len(self._current_graph.anchoring.anchors)
        response = self._graph_nav_client.upload_graph(
            graph=self._current_graph,
            generate_new_anchoring=true_if_empty,
        )
        # Upload the snapshots to the robot.
        for snapshot_id in response.unknown_waypoint_snapshot_ids:
            waypoint_snapshot = self._current_waypoint_snapshots[snapshot_id]
            self._graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
            print(f"Uploaded {waypoint_snapshot.id}")
        for snapshot_id in response.unknown_edge_snapshot_ids:
            edge_snapshot = self._current_edge_snapshots[snapshot_id]
            self._graph_nav_client.upload_edge_snapshot(edge_snapshot)
            print(f"Uploaded {edge_snapshot.id}")

        # The upload is complete! Check that the robot is localized to the graph,
        # and if it is not, prompt the user to localize the robot before attempting
        # any navigation commands.
        localization_state = self._graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            # The robot is not localized to the newly uploaded graph.
            print("\n")
            print(
                "Upload complete! The robot is currently not localized to the map; please localize",
                "the robot using commands (2) or (3) before attempting a navigation command.",
            )

    def _navigate_to_anchor(self, *args):
        """Navigate to a pose in seed frame, using anchors."""
        # The following options are accepted for arguments: [x, y], [x, y, yaw], [x, y, z, yaw],
        # [x, y, z, qw, qx, qy, qz].
        # When a value for z is not specified, we use the current z height.
        # When only yaw is specified, the quaternion is constructed from the yaw.
        # When yaw is not specified, an identity quaternion is used.

        if len(args) < 1 or len(args[0]) not in [2, 3, 4, 7]:
            print("Invalid arguments supplied.")
            return

        seed_T_goal = SE3Pose(float(args[0][0]), float(args[0][1]), 0.0, Quat())

        if len(args[0]) in [4, 7]:
            seed_T_goal.z = float(args[0][2])
        else:
            localization_state = self._graph_nav_client.get_localization_state()
            if not localization_state.localization.waypoint_id:
                print("Robot not localized")
                return
            seed_T_goal.z = localization_state.localization.seed_tform_body.position.z

        if len(args[0]) == 3:
            seed_T_goal.rot = Quat.from_yaw(float(args[0][2]))
        elif len(args[0]) == 4:
            seed_T_goal.rot = Quat.from_yaw(float(args[0][3]))
        elif len(args[0]) == 7:
            seed_T_goal.rot = Quat(
                w=float(args[0][3]),
                x=float(args[0][4]),
                y=float(args[0][5]),
                z=float(args[0][6]),
            )

        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        nav_to_cmd_id = None
        # Navigate to the destination.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to_anchor(
                    seed_T_goal.to_proto(),
                    1.0,
                    command_id=nav_to_cmd_id,
                )
            except ResponseError as e:
                print(f"Error while navigating {e}")
                break
            time.sleep(0.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)

        # Power off the robot if appropriate.
        if self._powered_on and not self._started_powered_on:
            # Sit the robot down + power off after the navigation command is complete.
            self.toggle_power(should_power_on=False)

    def _navigate_to_reuse(self, *args):
        """Navigate to a specific waypoint."""
        # Take the first argument as the destination waypoint.
        print(f"the args are {args}")
        if len(args) < 1:
            # If no waypoint id is given as input, then return without requesting navigation.
            print("No waypoint provided as a destination for navigate to.")
            return

        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0],
            self._current_graph,
            self._current_annotation_name_to_wp_id,
        )
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return
        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        nav_to_cmd_id = None
        # Navigate to the destination waypoint.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(
                    destination_waypoint,
                    1.0,
                    command_id=nav_to_cmd_id,
                )
            except ResponseError as e:
                print(f"Error while navigating {e}")
                break
            time.sleep(0.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)

    def _navigate_to(self, *args):
        """Navigate to a specific waypoint."""
        # Take the first argument as the destination waypoint.
        print(f"the args are {args}")
        if len(args) < 1:
            # If no waypoint id is given as input, then return without requesting navigation.
            print("No waypoint provided as a destination for navigate to.")
            return

        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0],
            self._current_graph,
            self._current_annotation_name_to_wp_id,
        )
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return
        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        nav_to_cmd_id = None
        # Navigate to the destination waypoint.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(
                    destination_waypoint,
                    1.0,
                    command_id=nav_to_cmd_id,
                )
            except ResponseError as e:
                print(f"Error while navigating {e}")
                break
            time.sleep(0.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)

        # Power off the robot if appropriate.
        if self._powered_on and not self._started_powered_on:
            # Sit the robot down + power off after the navigation command is complete.
            self.toggle_power(should_power_on=False)

    def _navigate_route(self, *args):
        """Navigate through a specific route of waypoints."""
        if len(args) < 1 or len(args[0]) < 1:
            # If no waypoint ids are given as input, then return without requesting navigation.
            print("No waypoints provided for navigate route.")
            return
        waypoint_ids = args[0]
        for i in range(len(waypoint_ids)):
            waypoint_ids[i] = graph_nav_util.find_unique_waypoint_id(
                waypoint_ids[i],
                self._current_graph,
                self._current_annotation_name_to_wp_id,
            )
            if not waypoint_ids[i]:
                # Failed to find the unique waypoint id.
                return

        edge_ids_list = []
        all_edges_found = True
        # Attempt to find edges in the current graph that match the ordered waypoint pairs.
        # These are necessary to create a valid route.
        for i in range(len(waypoint_ids) - 1):
            start_wp = waypoint_ids[i]
            end_wp = waypoint_ids[i + 1]
            edge_id = self._match_edge(self._current_edges, start_wp, end_wp)
            if edge_id is not None:
                edge_ids_list.append(edge_id)
            else:
                all_edges_found = False
                print("Failed to find an edge between waypoints: ", start_wp, " and ", end_wp)
                print(
                    "List the graph's waypoints and edges to ensure pairs of waypoints has an edge.",
                )
                break

        if all_edges_found:
            if not self.toggle_power(should_power_on=True):
                print("Failed to power on the robot, and cannot complete navigate route request.")
                return

            # Navigate a specific route.
            route = self._graph_nav_client.build_route(waypoint_ids, edge_ids_list)
            print("Route protobuf:", route)
            is_finished = False
            while not is_finished:
                # Issue the route command about twice a second such that it is easy to terminate the
                # navigation command (with estop or killing the program).
                nav_route_command_id = self._graph_nav_client.navigate_route(
                    route,
                    cmd_duration=1.0,
                )
                time.sleep(0.5)  # Sleep for half a second to allow for command execution.
                # Poll the robot for feedback to determine if the route is complete. Then sit
                # the robot down once it is finished.
                is_finished = self._check_success(nav_route_command_id)

            # Power off the robot if appropriate.
            if self._powered_on and not self._started_powered_on:
                # Sit the robot down + power off after the navigation command is complete.
                self.toggle_power(should_power_on=False)

    def _clear_graph(self, *args):
        """Clear the state of the map on the robot, removing all waypoints and edges."""
        return self._graph_nav_client.clear_graph()

    def toggle_power(self, should_power_on):
        """Power the robot on/off dependent on the current power state."""
        is_powered_on = self.check_is_powered_on()
        if not is_powered_on and should_power_on:
            # Power on the robot up before navigating when it is in a powered-off state.
            power_on(self._power_client)
            motors_on = False
            while not motors_on:
                future = self._robot_state_client.get_robot_state_async()
                state_response = future.result(
                    timeout=10,
                )  # 10 second timeout for waiting for the state response.
                if (
                    state_response.power_state.motor_power_state
                    == robot_state_pb2.PowerState.STATE_ON
                ):
                    motors_on = True
                else:
                    # Motors are not yet fully powered on.
                    time.sleep(0.25)
        elif is_powered_on and not should_power_on:
            # Safe power off (robot will sit then power down) when it is in a
            # powered-on state.
            safe_power_off(self._robot_command_client, self._robot_state_client)
        else:
            # Return the current power state without change.
            return is_powered_on
        # Update the locally stored power state.
        self.check_is_powered_on()
        return self._powered_on

    def check_is_powered_on(self):
        """Determine if the robot is powered on or off."""
        power_state = self._robot_state_client.get_robot_state().power_state
        self._powered_on = power_state.motor_power_state == power_state.STATE_ON
        return self._powered_on

    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot and sit when command succeeds."""
        if command_id == -1:
            # No command, so we have no status to check.
            return False
        status = self._graph_nav_client.navigation_feedback(command_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            # Successfully completed the navigation commands!
            return True
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            print("Robot got lost when navigating the route, the robot will now sit down.")
            return True
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            print("Robot got stuck when navigating the route, the robot will now sit down.")
            return True
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            print("Robot is impaired.")
            return True
        # Navigation command is not complete yet.
        return False

    def _match_edge(self, current_edges, waypoint1, waypoint2):
        """Find an edge in the graph that is between two waypoint ids."""
        # Return the correct edge id as soon as it's found.
        for edge_to_id in current_edges:
            for edge_from_id in current_edges[edge_to_id]:
                if (waypoint1 == edge_to_id) and (waypoint2 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint2, to_waypoint=waypoint1)
                if (waypoint2 == edge_to_id) and (waypoint1 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint1, to_waypoint=waypoint2)
        return None

    def _on_quit(self):
        """Cleanup on quit from the command line interface."""
        # Sit the robot down + power off after the navigation command is complete.
        if self._powered_on and not self._started_powered_on:
            self._robot_command_client.robot_command(
                RobotCommandBuilder.safe_power_off_command(),
                end_time_secs=time.time(),
            )

    def run(self):
        """Main loop for the command line interface."""
        while True:
            print(
                """
            Options:
            (1) Get localization state.
            (2) Initialize localization to the nearest fiducial (must be in sight of a fiducial).
            (3) Initialize localization to a specific waypoint (must be exactly at the waypoint)."""
                """
            (4) List the waypoint ids and edge ids of the map on the robot.
            (5) Upload the graph and its snapshots.
            (6) Navigate to. The destination waypoint id is the second argument.
            (7) Navigate route. The (in-order) waypoint ids of the route are the arguments.
            (8) Navigate to in seed frame. The following options are accepted for arguments: [x, y],
                [x, y, yaw], [x, y, z, yaw], [x, y, z, qw, qx, qy, qz]. (Don't type the braces).
                When a value for z is not specified, we use the current z height.
                When only yaw is specified, the quaternion is constructed from the yaw.
                When yaw is not specified, an identity quaternion is used.
            (9) Clear the current graph.
            (10) Navigate to all waypoints and collect sensor data
            (11) Compile edges from map into datastructure with costs
            (12) Get the pose of all waypoints in achor frame without moving
            (13) Download the graph (presumably the compressed one)
            (14) Compress graph
            (15) Add edges from compresssed graph to map
            (q) Exit.
            """,
            )
            try:
                inputs = input(">")
            except NameError:
                pass
            req_type = str.split(inputs)[0]

            if req_type == "q":
                self._on_quit()
                break

            if req_type not in self._command_dictionary:
                print("Request not in the known command dictionary.")
                continue
            try:
                cmd_func = self._command_dictionary[req_type]
                cmd_func(str.split(inputs)[1:])
            except Exception as e:
                print(e)


def main(argv):
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-u",
        "--upload-filepath",
        help="Full filepath to graph and snapshots to be uploaded.",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--data_directory_path",
        help="Full filepath to where images for waypoints should be stored.",
        required=False,
    )
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)

    # Setup and authenticate the robot.
    sdk = bosdyn.client.create_standard_sdk("GraphNavClient")
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)

    graph_nav_command_line = GraphNavInterface(
        robot,
        options.upload_filepath,
        options.data_directory_path,
    )
    lease_client = robot.ensure_client(LeaseClient.default_service_name)

    try:
        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            try:
                graph_nav_command_line.run()
                return True
            except Exception as exc:  # pylint: disable=broad-except
                print(exc)
                print("Graph nav command line client threw an error.")
                return False
    except ResourceAlreadyClaimedError:
        print(
            "The robot's lease is currently in use. Check for a tablet connection or try again in a few seconds.",
        )
        return False


if __name__ == "__main__":
    exit_code = 0
    if not main(sys.argv[1:]):
        exit_code = 1
    os._exit(exit_code)  # Exit hard, no cleanup that could block.
