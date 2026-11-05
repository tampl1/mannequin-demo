#!/bin/bash
exec > /workspace/ad_chain.log 2>&1
set -x
pip install --retries 10 --timeout 40 -q "diffusers>=0.35" "transformers>=4.51" accelerate imageio imageio-ffmpeg safetensors opencv-python-headless peft ftfy easydict
pip uninstall -y -q flash-attn 2>/dev/null
if [ ! -d /workspace/wan-code ]; then git clone -q --depth 1 https://github.com/Wan-Video/Wan2.2.git /workspace/wan-code; fi
sed -i 's/from .attention import flash_attention/from .attention import flash_attention, attention/' /workspace/wan-code/wan/modules/model.py
sed -i 's/x = flash_attention(/x = attention(/g' /workspace/wan-code/wan/modules/model.py
apt-get install -y -q libgl1 libglib2.0-0 ffmpeg 2>&1 | tail -1
cd /workspace
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 qwen_edit.py 2>&1 | tail -3
python3 wan_gen.py 2>&1 | tail -3
echo AD_CHAIN_DONE
