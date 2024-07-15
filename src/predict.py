
import os
import sys
# insert root directory into python module search path
sys.path.insert(1, os.getcwd())

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from model import UNetGenerator
from src.config import DATA_DIR
from utils import denormalize
from predict.transform_frames_to_video import transform_frames_in_video

# from torchvision.transforms import functional as F

# Assuming UNetGenerator is defined as above

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
        frame_files = [f for f in os.listdir(self.folder_path) if '_pose' in f]
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

        spectrogram = self.transform(spectrogram)

        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)

        # Normalize to [-1, 1]
        spectrogram = (spectrogram / 127.5) - 1

        return spectrogram, self.initial_frame


# Define the transformation


def predict(model_weights_file):
    # Load the model weights
    netG = UNetGenerator(input_channels=4, output_channels=3)
    netG.load_state_dict(torch.load(model_weights_file, map_location=torch.device('cpu')))
    netG.eval()  # Set the model to evaluation mode

    transform = transforms.Compose([
        transforms.RandomVerticalFlip(p=1),  # p=1.0 ensures the flip always happens
    ])

    # Create the dataset and dataloader for inference
    folder_path = os.path.join(DATA_DIR, "test")
    # TODO take a vide from youtube and transform it to spectrogram (with the code in train)
    #  place the inicial frame with the name _frame or _pose
    frames_dir = os.path.join(DATA_DIR, 'results/generated-frames')

    inference_dataset = SpectrogramNPYDataset(folder_path, transform=transform)
    inference_loader = DataLoader(inference_dataset, batch_size=16, shuffle=False)

    netG.train()  # Set the model to training mode

    # Move model to the correct device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netG.to(device)

    # Create a directory to save individual frames if it doesn't exist
    os.makedirs(f"{DATA_DIR}/results/generated-frames", exist_ok=True)

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
                        os.path.join(
                            frames_dir,
                            f'frame_{batch_idx * inference_loader.batch_size + i + 1}.png'))

                # Update the previous frame for the next iteration
                previous_frame = fake_frame


    # Path to save the video
    transform_frames_in_video(video_filename="output_video_8", frames_dir=frames_dir)


parser = argparse.ArgumentParser()
parser.add_argument("--model_weights_file", type=str, default="model_weights_Vframe.8.pth", help="Select a specific model weights file to inference")
args = parser.parse_args()

if __name__ == "__main__":
    predict(args.model_weights_file)