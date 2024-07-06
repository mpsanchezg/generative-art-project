# Use the SD model, generate the first image, and then a loop

from PIL import Image
import os
import torch

# Use the pre-trained SD pipeline, the poses images, and add the directory where we add the generated images
def generate_images(pipe, images, save_dir): 
    os.makedirs(save_dir, exist_ok=True)
    last_generated_image = None

    for i, pose_image in enumerate(images): #Loop for the images
        pose_image_pil = Image.fromarray(pose_image).convert("RGB")
        generator = torch.manual_seed(0)

        generated_image = pipe(
            prompt="", # Add text prompt
            negative_prompt="",
            num_inference_steps=20, # Add inference steps
            generator=generator,
            image=[last_generated_image, pose_image_pil], # Conditions the generation process on the last generated image and the current pose image
            controlnet_conditioning_scale=[0.6, 0.7] # Scales for the conditioning in order to apply controlnet. Check how it works exactly.
        ).images[0]

        output_path = os.path.join(save_dir, f'image_{i:03d}.png')
        generated_image.save(output_path)

        last_generated_image = generated_image
        print(f"Generated image for {i}th pose image and saved to {output_path}")