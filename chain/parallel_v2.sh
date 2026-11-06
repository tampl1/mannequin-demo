#!/bin/bash
exec > /workspace/shots.log 2>&1
set -x
export HF_HOME=/workspace/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=0
for i in 1 2 3; do python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download('Wan-AI/Wan2.2-TI2V-5B', allow_patterns=['*.json','Wan2.2_VAE.pth','models_t5_umt5-xxl-enc-bf16.pth','diffusion_pytorch_model-*.safetensors'])
print('weights:', p)
" && break || sleep 20; done
SNAP=$(python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Wan-AI/Wan2.2-TI2V-5B', allow_patterns=['*.json','Wan2.2_VAE.pth','models_t5_umt5-xxl-enc-bf16.pth','diffusion_pytorch_model-*.safetensors']))")
PREV="$SEED_FRAME"
for i in $SHOT_IDS; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /workspace/wan-code/generate.py --task ti2v-5B --size 1280*704 --ckpt_dir "$SNAP" --image "$PREV" --prompt "$PROMPT" --frame_num 121 --save_file /workspace/shot_$i --offload_model True --convert_model_dtype --t5_cpu || exit 1
  ffmpeg -y -v error -i /workspace/shot_$i.mp4 -vf "select='eq(n\,120)'" -frames:v 1 /workspace/shot_${i}_last.png
  PREV=/workspace/shot_${i}_last.png
done
echo SHOTS_DONE
