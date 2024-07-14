# Install the requirements

import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, ControlNetModel, AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import export_to_gif
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
from IPython.display import display, Image as displayImage
from PIL import Image
from typing import List
from diffusers.utils import export_to_video
from config import GENERATED_FRAMES_DIR, OUTPUT_VIDEO_DIR

from datetime import datetime

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)

motion_id = "guoyww/animatediff-motion-adapter-v1-5-2" # "CiaraRowles/TemporalDiff"
adapter = MotionAdapter.from_pretrained(motion_id)

model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    motion_adapter=adapter,
    controlnet=controlnet,
    vae=vae,
    custom_pipeline="pipeline_animatediff_controlnet",
).to(device="cuda", dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(
    model_id, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace", steps_offset=1, final_sigmas_type="sigma_min"
)

pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# Import the poses

folder_path = GENERATED_FRAMES_DIR
print(folder_path)
# List all npy files in the folder and take only the first 32
png_files = sorted(glob(os.path.join(folder_path, '*.png')))[:32]
print("len png_files", len(png_files))



# Change this in order to use .png files instead of .npy

# Check if png_files is not empty
if not png_files:
    raise ValueError("No png files found in the specified folder.")

# Load the npy files as images and convert them to PIL Images
openpose_frames = [Image.open(file) for file in png_files]
# Verify if the images were added successfully
print(f"Total frames: {len(openpose_frames)}")

prompt = "A dancer in a beautiful salon in Paris."
negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    # width=512,
    # height=768,
    width=384,
    height=512,
    num_frames=32,
    conditioning_frames=openpose_frames,
    num_inference_steps=20,
)

folder_path = os.path.join(OUTPUT_VIDEO_DIR, '{}-generated-frames'.format(datetime.now().strftime('%Y%m%d-%H')))

os.makedirs(folder_path, exist_ok=True)

export_to_gif(result.frames[0], os.path.join(folder_path, "result.gif"))
# display(displayImage("result.gif", embed=True))

# Define the directory path


# Define the full path for the video file
output_video_path = os.path.join(folder_path, 'output_video.mp4')
video_frames = result.frames[0]

# Set the desired frames per second (fps)
fps = 8  # Adjust this value as needed

# Export the video
video_path = export_to_video(video_frames, output_video_path, fps=fps)

print(f"Video saved at {video_path}")
