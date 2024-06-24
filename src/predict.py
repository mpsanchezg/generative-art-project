import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
# from torchvision.transforms import functional as F

import cv2

import torch.nn as nn
import random
import torch.nn.functional as F

from model import UNetGenerator
from utils import denormalize

# Assuming UNetGenerator is defined as above
# Load the model weights
netG = UNetGenerator(input_channels=4, output_channels=3)
netG.load_state_dict(torch.load('model_weights_V7.pth'))
netG.eval()  # Set the model to evaluation mode


# Dataset definition for inference
class SpectrogramNPYDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.npy_files = [f for f in os.listdir(folder_path) if '_spectrogram' in f]
        self.transform = transform  # Assign the transform function

        # Define the transformation for resizing
        self.resize_transform = transforms.Resize((256, 256))

        # Load initial frame
        self.initial_frame = self.load_initial_frame()

    def load_initial_frame(self):
        frame_files = [f for f in os.listdir(self.folder_path) if '_frame' in f]
        if len(frame_files) == 0:
            raise FileNotFoundError("No initial frame found in the folder.")
        frame_path = os.path.join(self.folder_path, frame_files[0])
        frame = np.load(frame_path)
        frame = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1)
        frame = torch.tensor(frame, dtype=torch.float32)
        frame = self.resize_transform(frame)  # Resize the frame to 256x256
        frame = (frame / 127.5) - 1  # Normalize to [-1, 1]

        return frame

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        npy_name = os.path.join(self.folder_path, self.npy_files[idx])
        spectrogram = np.load(npy_name)

        # Ensure it has the channel dimension
        spectrogram = torch.from_numpy(spectrogram.astype(np.float32)).unsqueeze(0)  # Add channel dimension

        spectrogram = transform(spectrogram)

        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)

        # Normalize to [-1, 1]
        spectrogram = (spectrogram / 127.5) - 1

        return spectrogram, self.initial_frame


# Define the transformation

transform = transforms.Compose([
    transforms.RandomVerticalFlip(p=1),  # p=1.0 ensures the flip always happens
])

# Create the dataset and dataloader for inference
folder_path = 'C:/Users/David/Documents/postgrau/Projecte/raining/frames/Inference Spectrograms'
inference_dataset = SpectrogramNPYDataset(folder_path, transform=transform)
inference_loader = DataLoader(inference_dataset, batch_size=16, shuffle=False)


# Visualize a few spectrograms and the initial frame
def visualize_spectrograms(dataset, num_samples=5):
    plt.figure(figsize=(15, 2.2))

    # Plot initial frame
    initial_frame = dataset.initial_frame.permute(1, 2, 0).numpy()
    initial_frame = (initial_frame + 1) * 127.5  # Denormalize
    plt.subplot(1, num_samples + 1, 1)
    plt.imshow(initial_frame.astype(np.uint8))
    plt.title('Initial Frame')

    for i in range(num_samples):
        spectrogram, _ = dataset[i]
        spectrogram = spectrogram.squeeze().numpy()
        plt.subplot(1, num_samples + 1, i + 2)
        plt.imshow(spectrogram, aspect='auto', origin='lower')
        plt.title(f'Sample {i + 1}')

    plt.show()


visualize_spectrograms(inference_dataset, num_samples=5)

# ---------------- second cell

netG.train()  # Set the model to training mode

# Move model to the correct device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netG.to(device)

# Create a directory to save individual frames if it doesn't exist
os.makedirs('results/individual_frames', exist_ok=True)

# Perform inference and save the generated frames
with torch.no_grad():
    # Load the initial frame
    fake_frame = inference_dataset.initial_frame.to(device).unsqueeze(0)

    # Process each batch in the inference loader
    for batch_idx, (spectrograms, initial_frame) in enumerate(inference_loader):
        spectrograms = spectrograms.to(device)  # Move spectrograms to the correct device

        # Initialize the previous frame with the initial frame
        previous_frame = fake_frame

        # Initialize hidden state for ConvLSTM
        hidden_state = None

        # Process each spectrogram in the batch
        for i, spectrogram in enumerate(spectrograms):
            # Generate fake frame
            fake_frame, hidden_state = netG(torch.cat((spectrogram.unsqueeze(0), previous_frame), dim=1), hidden_state)
            fake_frame_denorm = denormalize(fake_frame)  # Denormalize the generated frame

            # Save the generated frame
            save_image(fake_frame_denorm.squeeze(),
                       f'results/individual_frames/frame_{batch_idx * inference_loader.batch_size + i + 1}.png')

            # Update the previous frame for the next iteration
            previous_frame = fake_frame


# Visualize the first few results
def visualize_generated_frames(folder_path, num_samples=5):
    plt.figure(figsize=(15, 2.7))
    for i in range(num_samples):
        frame_path = os.path.join(folder_path, f'frame_{i + 1}.png')
        frame = plt.imread(frame_path)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(frame)
        plt.title(f'Frame {i + 1}')
        plt.axis('off')
    plt.show()


visualize_generated_frames('results/individual_frames', num_samples=5)

# -------------------- third cell


# Directory containing the frames
frames_dir = 'results/individual_frames'

# Path to save the video
video_path = 'results/output_video_V10_5.mp4'

# Frames per second
fps = 30

# Get the list of all frame files and sort them numerically
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')],
                     key=lambda x: int(x.split('_')[1].split('.')[0]))

# Read the first frame to get the dimensions
first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
height, width, layers = first_frame.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

# Write each frame to the video
for frame_file in frame_files:
    frame_path = os.path.join(frames_dir, frame_file)
    frame = cv2.imread(frame_path)
    video.write(frame)

# Release the VideoWriter object
video.release()

print(f'Video saved at {video_path}')



