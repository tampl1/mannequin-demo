#!/bin/bash
exec > /workspace/chained30.log 2>&1
set -x
export HF_HOME=/workspace/hf-cache
apt-get update -qq && apt-get install -y -q ffmpeg libgl1 libglib2.0-0
pip install --retries 20 --timeout 60 -q "diffusers>=0.35" transformers accelerate imageio imageio-ffmpeg safetensors einops ftfy easydict decord librosa
pip uninstall -y -q flash-attn 2>/dev/null
git clone -q --depth 1 https://github.com/Wan-Video/Wan2.2.git /workspace/wan-code
sed -i 's/from .attention import flash_attention/from .attention import flash_attention, attention/' /workspace/wan-code/wan/modules/model.py
sed -i 's/x = flash_attention(/x = attention(/g' /workspace/wan-code/wan/modules/model.py
for i in 1 2 3; do python3 -c "
from huggingface_hub import snapshot_download
print('weights:', snapshot_download('Wan-AI/Wan2.2-TI2V-5B'))" && break || sleep 20; done
SNAP=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Wan-AI/Wan2.2-TI2V-5B'))")
cp /workspace/input_human.png /workspace/shot_0_last.png
declare -a PROMPTS
PROMPTS[1]="She lifts the elegant comb from the table and slowly runs it through her long hair, smoothing it, relaxed happy expression, studio lighting, smooth cinematic motion, photorealistic"
PROMPTS[2]="She holds the comb up to the camera showing its teeth, rotating it slightly, confident warm smile, studio lighting, smooth cinematic motion, photorealistic"
PROMPTS[3]="She combs her hair once more slowly, then nods warmly at the camera, hair shining, studio lighting, smooth cinematic motion, photorealistic"
PROMPTS[4]="She places the comb gently on the table and folds her hands, looking at the camera with a friendly final smile, studio lighting, smooth cinematic motion, photorealistic"
PREV=/workspace/shot_0_last.png
for i in 1 2 3 4; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /workspace/wan-code/generate.py --task ti2v-5B --size 1280*704 --ckpt_dir "$SNAP" --image "$PREV" --prompt "${PROMPTS[$i]}" --frame_num 121 --save_file /workspace/shot_$i --offload_model True --convert_model_dtype --t5_cpu || exit 1
  ffmpeg -y -v error -i /workspace/shot_$i.mp4 -vf "select='eq(n\,120)'" -frames:v 1 /workspace/shot_${i}_last.png
  PREV=/workspace/shot_${i}_last.png
done
printf "file '/workspace/shot_0.mp4'\n" > /tmp/cl.txt
for i in 0 1 2 3 4; do :; done
cat > /workspace/concat.txt << 'CL'
file '/workspace/human_ad.mp4'
file '/workspace/shot_1.mp4'
file '/workspace/shot_2.mp4'
file '/workspace/shot_3.mp4'
file '/workspace/shot_4.mp4'
CL
ffmpeg -y -v error -f concat -safe 0 -i /workspace/concat.txt -c copy /workspace/ad30.mp4
echo CHAINED30_DONE
