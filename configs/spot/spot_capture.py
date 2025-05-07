base_dir = "./experiments/Spot_Captures"  # Root Directory to Save iPhone Dataset
scene_name = "spot_capture"  # Scan Name (seems to be used for folder naming)
num_frames = 48  # Desired number of frames to capture

full_res_width = 640
full_res_height = 480
downscale_factor = 1.0
densify_downscale_factor = 2.0  # Default: 4.0

map_every = 1
keyframe_every = 5 if num_frames >= 25 else int(num_frames // 5)
mapping_window_size = 24  # Default: 32 - Probably affects VRAM
tracking_iters = 60  # Default: 60
mapping_iters = 60  # Default: 60


config = {
    "workdir": f"./{base_dir}/{scene_name}",
    "run_name": "Spot_Capture",
    "overwrite": False,  # Rewrite over dataset if it exists?
    "depth_scale": 10.0,  # Depth Scale used when saving depth
    "num_frames": num_frames,
    "seed": 0,
    "primary_device": "cuda:0",
    "map_every": map_every,  # Mapping every nth frame
    "keyframe_every": keyframe_every,  # Keyframe every nth frame
    "mapping_window_size": mapping_window_size,  # Mapping window size
    "report_global_progress_every": 100,  # Report Global Progress every nth frame
    "eval_every": 1,  # Evaluate every nth frame (at end of SLAM)
    "scene_radius_depth_ratio": 3,  # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    "mean_sq_dist_method": "projective",  # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    "gaussian_distribution": "isotropic",  # ["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)
    "report_iter_progress": False,
    "load_checkpoint": False,
    "checkpoint_time_idx": 130,
    "save_checkpoints": False,  # Save Checkpoints
    "checkpoint_interval": 5,  # Checkpoint Interval
    "use_wandb": False,
    "data": {
        "dataset_name": "nerfcapture",
        "basedir": base_dir,
        "sequence": scene_name,
        "desired_image_height": int(full_res_height // downscale_factor),
        "desired_image_width": int(full_res_width // downscale_factor),
        "densification_image_height": int(full_res_height // densify_downscale_factor),
        "densification_image_width": int(full_res_width // densify_downscale_factor),
        "start": 0,
        "end": -1,
        "stride": 1,
        "num_frames": num_frames,
    },
    "tracking": {
        "use_gt_poses": True,  # Use GT Poses for Tracking
        "forward_prop": True,  # Forward Propagate Poses
        "visualize_tracking_loss": False,  # Visualize Tracking Diff Images
        "num_iters": tracking_iters,
        "use_sil_for_loss": True,
        "sil_thres": 0.99,
        "use_l1": True,
        "use_depth_loss_thres": True,
        "depth_loss_thres": 20000,  # Num of Tracking Iters becomes twice if this value is not met
        "ignore_outlier_depth_loss": False,
        "use_uncertainty_for_loss_mask": False,
        "use_uncertainty_for_loss": False,
        "use_chamfer": False,
        "loss_weights": {
            "im": 0.5,
            "depth": 1.0,
        },
        "lrs": {
            "means3D": 0.0,
            "rgb_colors": 0.0,
            "unnorm_rotations": 0.0,
            "logit_opacities": 0.0,
            "log_scales": 0.0,
            "cam_unnorm_rots": 0.001,
            "cam_trans": 0.004,
        },
    },
    "mapping": {
        "num_iters": mapping_iters,
        "add_new_gaussians": True,
        "sil_thres": 0.5,  # For Addition of new Gaussians
        "use_l1": True,
        "ignore_outlier_depth_loss": False,
        "use_sil_for_loss": False,
        "use_uncertainty_for_loss_mask": False,
        "use_uncertainty_for_loss": False,
        "use_chamfer": False,
        "loss_weights": {
            "im": 0.5,
            "depth": 1.0,
        },
        "lrs": {
            "means3D": 0.0001,
            "rgb_colors": 0.0025,
            "unnorm_rotations": 0.001,
            "logit_opacities": 0.05,
            "log_scales": 0.001,
            "cam_unnorm_rots": 0.0000,
            "cam_trans": 0.0000,
        },
        "prune_gaussians": True,  # Prune Gaussians during Mapping
        "pruning_dict": {  # Needs to be updated based on the number of mapping iterations
            "start_after": 0,
            "remove_big_after": 0,
            "stop_after": 20,
            "prune_every": 20,
            "removal_opacity_threshold": 0.005,
            "final_removal_opacity_threshold": 0.005,
            "reset_opacities": False,
            "reset_opacities_every": 500,  # Doesn't consider iter 0
        },
        "use_gaussian_splatting_densification": False,  # Use Gaussian Splatting-based Densification during Mapping
        "densify_dict": {  # Needs to be updated based on the number of mapping iterations
            "start_after": 500,
            "remove_big_after": 3000,
            "stop_after": 5000,
            "densify_every": 100,
            "grad_thresh": 0.0002,
            "num_to_split_into": 2,
            "removal_opacity_threshold": 0.005,
            "final_removal_opacity_threshold": 0.005,
            "reset_opacities_every": 3000,  # Doesn't consider iter 0
        },
    },
    "viz": {
        "render_mode": "color",  # ['color', 'depth' or 'centers']
        "offset_first_viz_cam": True,  # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        "show_sil": False,  # Show Silhouette instead of RGB
        "visualize_cams": True,  # Visualize Camera Frustums and Trajectory
        "viz_w": 600,
        "viz_h": 340,
        "viz_near": 0.01,
        "viz_far": 100.0,
        "view_scale": 2,
        "viz_fps": 5,  # FPS for Online Recon Viz
        "enter_interactive_post_online": False,  # Enter Interactive Mode after Online Recon Viz
    },
}
