#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <video_file>"
    exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ../D-PoSE/

source ~/.bashrc
source venv/bin/activate

VIDEO="$(realpath "$1")"
CSV_OUT="${VIDEO}_3DBody.csv"
TMP_OUT=$(mktemp -d)

python3 demo_webcam_csv.py --input "$VIDEO" --output_folder "$TMP_OUT"

mv "$TMP_OUT/3DPoints.csv" "$CSV_OUT"
rm -rf "$TMP_OUT"

echo "Saved to: $CSV_OUT"
