import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
img = load_image("/workspace/input_human.png").resize((1024, 1024))
out = pipe(image=[img],
    prompt="Add a dark tailored blazer over her outfit. Place a hair comb in her right hand. Keep her face, hair and the background unchanged. Photorealistic.",
    num_inference_steps=28, guidance_scale=4.0,
    generator=torch.Generator("cuda").manual_seed(42)).images[0]
out = out.resize((1280, 704))
out.save("/workspace/human_edited.png")
print("EDIT_DONE")
