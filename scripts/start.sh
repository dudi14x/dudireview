#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export YOLO_CONFIG_DIR="$PWD/data/yolo"
exec .venv/bin/python -m floorreview.app
