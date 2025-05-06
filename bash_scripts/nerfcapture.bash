#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: bash_scripts/nerfcapture.bash <config_file>"
    exit
fi

if [ ! -f $1 ]; then
    echo "Config file not found!"
    exit
fi

echo "[uv sync] ensuring venv is up to date..."
uv sync

echo "[dataset] running nerfcapture-dataset..."
uv run --env-file .env scripts/nerfcapture2dataset.py --config $1

echo "[slam] running SplaTAM..."
uv run scripts/splatam.py $1

echo "[viz] visualizing output..."
uv run viz_scripts/final_recon.py $1