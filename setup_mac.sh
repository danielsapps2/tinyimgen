#!/usr/bin/env bash
# One-time setup on an Apple-silicon Mac (M1 or newer, macOS 14+). See README: "Apple silicon".
#
#   ./setup_mac.sh              # image server (.venv) + 3D tools (.venv3d), and the FLUX.2 klein weights (~16 GB)
#   ./setup_mac.sh --zimage     # also download Z-Image Turbo (~25 GB more; needs a 48 GB+ Mac to run)
#   ./setup_mac.sh --local3d    # also install the local Hunyuan3D-2 engine (Tripo on fal is the default)
set -euo pipefail
cd "$(dirname "$0")"

ZIMAGE=0; LOCAL3D=0
for a in "$@"; do
  case "$a" in
    --zimage) ZIMAGE=1 ;;
    --local3d) LOCAL3D=1 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This script is for Apple-silicon Macs. On Linux + NVIDIA, see README: Environment."; exit 1
fi

if ! command -v uv >/dev/null; then
  echo "Installing uv (Python package manager)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "== .venv: image server (FLUX.2 klein on Metal)"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  "torch==2.11.0" "torchvision==0.26.0" \
  "diffusers==0.41.0" "transformers==5.19.0" "accelerate==1.15.0" "huggingface-hub==1.33.0" \
  "safetensors" "sentencepiece" "peft" \
  "fastapi" "uvicorn" "pydantic" "pillow" "numpy" "fal-client==1.0.3"

echo "== .venv3d: 3D tools (Tripo export, rigging, animation)"
uv venv --python 3.12 .venv3d
uv pip install --python .venv3d/bin/python \
  "torch==2.11.0" "torchvision==0.26.0" \
  "transformers==4.57.1" "diffusers==0.35.2" "accelerate==1.15.0" "huggingface-hub==0.36.2" \
  "numpy" "scipy" "scikit-image" "trimesh==5.1.1" "pymeshlab" "xatlas" "fast-simplification" "pygltflib" \
  "mediapipe==1.1.0" "onnxruntime" "rembg" "einops" "omegaconf" "fal-client==1.0.3" "pillow"

if [[ $LOCAL3D == 1 ]]; then
  echo "== Hunyuan3D-2 (local 3D engine)"
  if [[ ! -d third_party/Hunyuan3D-2 ]]; then
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2 third_party/Hunyuan3D-2
    git -C third_party/Hunyuan3D-2 checkout f8db630
  fi
  # --no-deps: its requirements pull gradio and CUDA build tools that the shape model doesn't need
  uv pip install --python .venv3d/bin/python --no-deps -e third_party/Hunyuan3D-2
fi

echo "== Model weights (the server runs offline, so they are fetched now)"
.venv/bin/python - "$ZIMAGE" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download("black-forest-labs/FLUX.2-klein-4B")
if sys.argv[1] == "1":
    snapshot_download("Tongyi-MAI/Z-Image-Turbo")
EOF

cat <<'EOF'

Done. Start it with:
    .venv/bin/python server.py            # http://127.0.0.1:7860
The 3D tab needs a fal API key in ~/.config/tinyimgen/fal_key (or FAL_KEY).
For FBX export, install Blender (https://www.blender.org/download/) into /Applications.
EOF
