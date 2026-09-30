#!/bin/bash
# uv-based install of the MMBench/Newt dependency set (docker/environment.yaml
# pip section), into a node-local venv under $TMP. No conda.
set -eo pipefail

UV="$HOME/.local/bin/uv"
BASE=/tmp/yn030245
VENV=$BASE/newt-venv
export UV_CACHE_DIR=$BASE/uv-cache
export TMPDIR=$BASE/tmp
mkdir -p "$BASE" "$UV_CACHE_DIR" "$TMPDIR"

# swig is needed as a *binary* to build box2d-py; module version beats the wheel.
module load SWIG/4.3.1 2>/dev/null || true
echo "swig: $(command -v swig || echo 'none yet')"

echo "=== uv $($UV --version) ; target $VENV ==="
rm -rf "$VENV"
"$UV" venv --python 3.10 "$VENV"
PY="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"
"$PY" -V

echo
echo "=== phase 1/4: build backends (swig for box2d, pinned numpy) ==="
"$UV" pip install --python "$PY" "setuptools<70" wheel "swig==4.3.1" "numpy==1.24.4"

echo
echo "=== phase 2/4: main solve (torch 2.8, mujoco, ManiSkill, Meta-World) ==="
"$UV" pip install --python "$PY" \
  "numpy==1.24.4" \
  "torch==2.8.0" "torchvision==0.23.0" "torchrl==0.10.0" "tensordict==0.10.0" \
  "dm-control==1.0.34" "mujoco==3.3.6" "glfw==2.10.0" \
  "gymnasium==0.29.1" "gymnasium[box2d]" \
  "gpustat==1.1.1" "ffmpeg==1.4" \
  "imageio==2.37.0" "imageio-ffmpeg==0.6.0" "h5py==3.14.0" \
  "hydra-core==1.3.2" "hydra-submitit-launcher==1.2.0" "submitit==1.5.3" "omegaconf==2.3.0" \
  "git+https://github.com/nicklashansen/metaworld.git" \
  "mani_skill-nightly==2025.9.19.39" \
  "moviepy==1.0.3" "robodesk==1.0.0" "ogbench==1.1.5" \
  "transformers==4.56.2" "kornia==0.8.1" \
  "termcolor==3.1.0" "tqdm==4.67.1" "wandb==0.22.1"

echo
echo "=== phase 3/4: ale_py==0.10 (README post-step; --no-deps so it does not"
echo "    drag gymnasium off the pinned 0.29.1) ==="
"$UV" pip install --python "$PY" --no-deps "ale_py==0.10"

echo
echo "=== phase 4/4: snap numpy back to the pinned 1.24.4 ==="
"$UV" pip install --python "$PY" --no-deps "numpy==1.24.4"

echo
"$UV" pip freeze --python "$PY" > "$HOME/newt/newt_pip_freeze.txt"
echo "PACKAGES=$(wc -l < "$HOME/newt/newt_pip_freeze.txt")"
du -sh "$VENV"
echo "BUILD_OK"
