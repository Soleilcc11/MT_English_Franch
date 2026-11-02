#!/usr/bin/env bash
# Run on the Linux RTX 4090 host, after copying this project to its data disk.
# Installation and GPU discovery only; this script does not start training.
set -euo pipefail

dry_run=0
case "${1:-}" in
  --dry-run) dry_run=1 ;;
  --help|-h)
    printf '%s\n' 'Usage: bash scripts/setup_4090_torch27.sh [--dry-run]' \
      'ENFR_PYTHON may select the Python 3.11 interpreter (default: python3).' \
      'Creates .venv-torch27 and .venv-tf215 inside the project.'
    exit 0 ;;
  '') ;;
  *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then
  printf '%s\n' 'At most one argument is supported.' >&2
  exit 2
fi

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
enfr_python="${ENFR_PYTHON:-python3}"
torch_python=".venv-torch27/bin/python"
tf_python=".venv-tf215/bin/python"

run() {
  if [ "$dry_run" -eq 1 ]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [ "$dry_run" -eq 0 ]; then
  "$enfr_python" -c 'import platform, sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 required"; assert platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64"), "Linux x86_64 / WSL2 required"'
  command -v nvidia-smi >/dev/null
  nvidia-smi
  # Do not reuse an arbitrary folder or a Conda environment as a venv.
  for env_dir in .venv-torch27 .venv-tf215; do
    if [ -e "$env_dir" ] && { [ ! -f "$env_dir/pyvenv.cfg" ] || [ ! -x "$env_dir/bin/python" ]; }; then
      printf 'Existing path is not a complete Python venv: %s\n' "$env_dir" >&2
      exit 1
    fi
    if [ -x "$env_dir/bin/python" ]; then
      "$env_dir/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Existing venv must use Python 3.11"'
    fi
  done
fi

run "$enfr_python" -m venv .venv-torch27
run "$enfr_python" -m venv .venv-tf215
run "$torch_python" -m pip install --no-cache-dir --upgrade pip
run "$tf_python" -m pip install --no-cache-dir --upgrade pip
run "$torch_python" -m pip install --no-cache-dir 'torch==2.7.0+cu128' --index-url https://download.pytorch.org/whl/cu128
run "$torch_python" -m pip install --no-cache-dir -r scripts/requirements-torch27.txt
run "$tf_python" -m pip install --no-cache-dir -r scripts/requirements-tensorflow.txt
run "$torch_python" -m pip check
run "$tf_python" -m pip check
run "$torch_python" -c 'import torch; assert torch.__version__ == "2.7.0+cu128", torch.__version__; assert torch.version.cuda == "12.8", torch.version.cuda'
run "$torch_python" scripts/run.py doctor --framework torch --require-gpu --output output/host4090-torch27/doctor-torch.json
run "$tf_python" scripts/run.py doctor --framework tensorflow --require-gpu --output output/host4090-torch27/doctor-tensorflow.json

if [ "$dry_run" -eq 0 ]; then
  nvidia-smi > output/host4090-torch27/nvidia-smi.txt
  "$torch_python" -m pip freeze > output/host4090-torch27/torch-environment.lock.txt
  "$tf_python" -m pip freeze > output/host4090-torch27/tensorflow-environment.lock.txt
  printf '%s\n' 'Environments installed and GPUs detected. Next run parity, tests and both profiles as documented in README.md.'
fi
