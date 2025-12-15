#!/usr/bin/env bash
set -euo pipefail

# Automated setup script to install Miniforge (if needed), create a conda env
# and install TensorFlow + project requirements.
# Usage: ./scripts/setup_conda_env.sh [env-name] [python-version]
# Example: ./scripts/setup_conda_env.sh trading 3.11

ENV_NAME="${1:-trading}"
PY_VERSION="${2:-3.11}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REQ_FILE="$REPO_ROOT/requirements.txt"

echo "Repository root: $REPO_ROOT"
echo "Environment name: $ENV_NAME"
echo "Python version: $PY_VERSION"

conda_exists() { command -v conda >/dev/null 2>&1; }

install_miniforge() {
  echo "Conda not found. Installing Miniforge..."
  case "$(uname -s)" in
    Linux*) URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" ;;
    Darwin*)
      if [[ "$(uname -m)" == "arm64" ]]; then
        URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
      else
        URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
      fi
      ;;
    *) echo "Unsupported OS: $(uname -s). Please install Miniforge/Miniconda manually."; exit 1 ;;
  esac

  TMP="$(mktemp)"
  echo "Downloading: $URL"
  curl -L "$URL" -o "$TMP"
  bash "$TMP" -b -p "$HOME/miniforge3"
  rm "$TMP"
  export PATH="$HOME/miniforge3/bin:$PATH"
  echo "Miniforge installed to $HOME/miniforge3"
}

if ! conda_exists; then
  install_miniforge
else
  echo "Conda detected: $(conda --version)"
fi

echo "Creating conda environment '$ENV_NAME' with python=$PY_VERSION..."
conda create -n "$ENV_NAME" python="$PY_VERSION" -c conda-forge -y

echo "Activating environment and installing TensorFlow (conda-forge)..."
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
conda install -c conda-forge tensorflow -y

echo "Upgrading pip and installing remaining requirements from $REQ_FILE..."
python -m pip install --upgrade pip
if [[ -f "$REQ_FILE" ]]; then
  pip install -r "$REQ_FILE"
else
  echo "Warning: requirements.txt not found at $REQ_FILE"
fi

echo "Setup complete. To use the environment run: conda activate $ENV_NAME"
echo "If you're on Windows, consider using PowerShell commands in README or run this script inside WSL/Git-Bash."
