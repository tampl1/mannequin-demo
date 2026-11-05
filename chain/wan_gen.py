import sys, os, subprocess
from huggingface_hub import snapshot_download
snap = snapshot_download('Wan-AI/Wan2.2-TI2V-5B')
env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
cmd = ["python3", "/workspace/wan-code/generate.py", "--task", "ti2v-5B", "--size", "1280*704",
       "--ckpt_dir", snap, "--image", "/workspace/human_edited.png",
       "--prompt", "The young woman in the dark blazer combs her hair with the comb in her right hand, then raises the comb showing it to the camera with a friendly smile and advertises it, natural studio lighting, smooth cinematic motion, photorealistic",
       "--frame_num", "121", "--save_file", "/workspace/human_ad.mp4",
       "--offload_model", "True", "--convert_model_dtype", "--t5_cpu"]
subprocess.run(cmd, check=True, env=env)
print("AD_DONE")
