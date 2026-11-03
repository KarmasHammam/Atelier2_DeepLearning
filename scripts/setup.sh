#!/usr/bin/env bash
# Installation locale CPU ; pour Colab GPU, utiliser les cellules des notebooks.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check torch==2.14.1 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
.venv/bin/python -c "from src.data import mnist; mnist()"
