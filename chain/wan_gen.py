import sys, torch
from diffusers.utils import export_to_video, load_image
from diffusers import Wan22Pipeline
try:
    from diffusers import Wan22Pipeline as WanPipe
except ImportError:
    from diffusers import WanPipeline as WanPipe
snap = sys.argv[1]
pipe = WanPipe.from_pretrained(snap, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
img = load_image("/workspace/human_edited.png").resize((1280, 704))
out = pipe(image=img,
    prompt="The young woman in the dark blazer combs her hair with the comb in her right hand, then raises the comb showing it to the camera with a friendly smile and advertises it, natural studio lighting, smooth cinematic motion, photorealistic",
    negative_prompt="static, stiff, distorted face, deformed hands, extra limbs, blurry, low quality, text, watermark",
    height=704, width=1280, num_frames=121, guidance_scale=5.0, num_inference_steps=30,
    generator=torch.Generator("cuda").manual_seed(42)).frames[0]
export_to_video(out, "/workspace/human_ad.mp4", fps=24)
print("AD_DONE")
