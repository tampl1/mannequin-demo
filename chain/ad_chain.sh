#!/bin/bash
exec > /workspace/ad_chain.log 2>&1
set -x
apt-get update -qq && apt-get install -y -q ffmpeg libgl1 libglib2.0-0
pip install --retries 10 --timeout 40 -q "diffusers>=0.35" "transformers>=4.51" accelerate imageio imageio-ffmpeg safetensors opencv-python-headless
pip uninstall -y -q flash-attn 2>/dev/null
cd /workspace
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 qwen_edit.py 2>&1 | tail -2
SNAP=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Wan-AI/Wan2.2-TI2V-5B'))")
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 wan_gen.py "$SNAP" 2>&1 | tail -2
echo AD_CHAIN_DONE
