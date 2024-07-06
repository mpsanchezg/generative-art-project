# model_loading.py
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

# Paths to the TemporalNet models
controlnet1_path = "CiaraRowles/controlnet-temporalnet-sdxl-1.0"
controlnet2_path = "thibaud/controlnet-openpose-sdxl-1.0"

# Load the TemporalNet models
controlnet = [
    ControlNetModel.from_pretrained(controlnet1_path, torch_dtype=torch.float16),
    ControlNetModel.from_pretrained(controlnet2_path, torch_dtype=torch.float16)
]

# Load the pipeline with TemporalNet models
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()