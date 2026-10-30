#!/bin/bash
set -euo pipefail

mkdir -p data/external

URL="https://datahub.io/core/openml-datasets/_r/-/data/mammography/mammography.csv"
RAW="data/external/mammography_raw.csv"

if [ ! -f "$RAW" ]; then
    echo "Downloading Mammography dataset..."
    curl -fL --retry 2 "$URL" -o "$RAW"
else
    echo "Using existing Mammography raw dataset."
fi

echo "Preparing dataset for OptIForest..."
"${PYTHON:-python3}" scripts/prepare_mammography.py

echo "Mammography setup completed."
