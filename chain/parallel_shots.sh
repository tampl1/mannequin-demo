#!/bin/bash
exec > /workspace/shots.log 2>&1
set -x
apt-get update -qq && apt-get install -y -q ffmpeg libgl1 libglib2.0-0
pip install --retries 20 --timeout 60 -q "diffusers>=0.35" transformers accelerate imageio imageio-ffmpeg safetensors einops ftfy easydict decord librosa
pip uninstall -y -q flash-attn 2>/dev/null
if [ ! -d /workspace/wan-code ]; then git clone -q --depth 1 https://github.com/Wan-Video/Wan2.2.git /workspace/wan-code; fi
sed -i 's/from .attention import flash_attention/from .attention import flash_attention, attention/' /workspace/wan-code/wan/modules/model.py
sed -i 's/x = flash_attention(/x = attention(/g' /workspace/wan-code/wan/modules/model.py
export HF_HOME=/workspace/hf-cache
for i in 1 2 3; do python3 -c "
from huggingface_hub import snapshot_download
print('weights:', snapshot_download('Wan-AI/Wan2.2-TI2V-5B'))" && break || sleep 20; done
SNAP=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Wan-AI/Wan2.2-TI2V-5B'))")
PREV="$SEED_FRAME"
for i in $SHOT_IDS; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /workspace/wan-code/generate.py --task ti2v-5B --size 1280*704 --ckpt_dir "$SNAP" --image "$PREV" --prompt "${PROMPT}" --frame_num 121 --save_file /workspace/shot_$i --offload_model True --convert_model_dtype --t5_cpu || exit 1
  ffmpeg -y -v error -i /workspace/shot_$i.mp4 -vf "select='eq(n\,120)'" -frames:v 1 /workspace/shot_${i}_last.png
  PREV=/workspace/shot_${i}_last.png
done
echo SHOTS_DONE
