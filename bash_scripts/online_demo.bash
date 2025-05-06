#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: bash_scripts/online_demo.bash <config_file>"
    exit
fi

if [ ! -f $1 ]; then
    echo "Config file not found!"
    exit
fi

echo "[uv sync] ensuring venv is up to date..."
uv sync

# Online Dataset Capture & SplaTAM
uv run scripts/iphone_demo.py --config $1

# Visualize SplaTAM Output
uv run viz_scripts/final_recon.py $1
