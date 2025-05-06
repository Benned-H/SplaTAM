#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: bash_scripts/nerfcapture2dataset.bash <config_file>"
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