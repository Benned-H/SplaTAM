# setup.py
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Path helper for extension sources
here = os.path.abspath(os.path.dirname(__file__))

setup(
    name="splatam",
    version="0.0.1",
    package_dir={
        "diff_gaussian_rasterization": "diff-gaussian-rasterization-w-depth/diff_gaussian_rasterization",
    },
    packages=find_packages(
        exclude=[
            "assets",
            "docker",
            "configs",
            "datasets",
            "experiments",
            "viz_scripts",
            "bash_scripts",
        ],
    ),
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "tqdm",
        "Pillow",
        "opencv-python",
        "imageio",
        "matplotlib",
        "kornia",
        "natsort",
        "pyyaml",
        "wandb",
        "lpips",
        "open3d",
        "torchmetrics",
        "cyclonedds",
        "pytorch-msssim",
        "plyfile",
        "faiss-gpu",
    ],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                os.path.join(
                    "diff-gaussian-rasterization-w-depth",
                    "cuda_rasterizer",
                    "rasterizer_impl.cu",
                ),
                os.path.join(
                    "diff-gaussian-rasterization-w-depth",
                    "cuda_rasterizer",
                    "forward.cu",
                ),
                os.path.join(
                    "diff-gaussian-rasterization-w-depth",
                    "cuda_rasterizer",
                    "backward.cu",
                ),
                os.path.join("diff-gaussian-rasterization-w-depth", "rasterize_points.cu"),
                os.path.join("diff-gaussian-rasterization-w-depth", "ext.cpp"),
            ],
            include_dirs=[
                os.path.join(here, "diff-gaussian-rasterization-w-depth", "third_party", "glm"),
            ],
            extra_compile_args={
                "cxx": ["-std=c++17"],  # Host compiler
                "nvcc": ["-std=c++17", "-Ithird_party/glm/"],  # NVCC
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
