#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPOSE_DIR="$SCRIPT_DIR/D-PoSE"
VENV_DIR="$DPOSE_DIR/venv"

echo "=== D-PoSE Setup ==="

# Clone D-PoSE webcam branch if not present
if [ -d "$DPOSE_DIR/.git" ]; then
    echo "D-PoSE already cloned — pulling latest..."
    git -C "$DPOSE_DIR" pull
else
    if [ -e "$DPOSE_DIR" ]; then
        echo "ERROR: '$DPOSE_DIR' exists but is not a git repository (it may be a symlink)."
        echo "Remove or rename it first, then re-run this script."
        exit 1
    fi
    echo "Cloning D-PoSE (webcam branch)..."
    git clone --branch webcam https://github.com/AmmarkoV/D-PoSE "$DPOSE_DIR"
fi

# Create venv inside D-PoSE
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

# Activate and install requirements
echo "Installing Python requirements..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$DPOSE_DIR/requirements.txt"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the environment for development:"
echo "  source D-PoSE/venv/bin/activate"
echo ""
echo "To run the webcam demo:"
echo "  source /opt/ros/humble/setup.bash"
echo "  source D-PoSE/venv/bin/activate"
echo "  python3 ros_demo_webcam.py"
