# Custom transformations

import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from utils import centralize_pose


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class TimeMasking(object):
    def __init__(self, max_mask_size):
        self.max_mask_size = max_mask_size

    def __call__(self, tensor):
        time_mask_size = np.random.randint(0, self.max_mask_size)
        time_mask_start = np.random.randint(0, tensor.size(2) - time_mask_size)
        tensor[:, :, time_mask_start:time_mask_start + time_mask_size] = 0
        return tensor


class FrequencyMasking(object):
    def __init__(self, max_mask_size):
        self.max_mask_size = max_mask_size

    def __call__(self, tensor):
        freq_mask_size = np.random.randint(0, self.max_mask_size)
        freq_mask_start = np.random.randint(0, tensor.size(1) - freq_mask_size)
        tensor[:, freq_mask_start:freq_mask_start + freq_mask_size, :] = 0
        return tensor


# Dataset class
# Dataset class
class FrameSpectrogramDataset(Dataset):
    def __init__(self, root_dir, video_selected=None, sr=22050, n_fft=2048, n_mels=128):
        self.root_dir = root_dir
        self.video_selected = video_selected

        if (video_selected == None):
            self.frame_files = sorted(
                [f for f in os.listdir(root_dir) if '_pose' in f and
                os.path.isfile(os.path.join(root_dir, f))])
        else:
            self.frame_files = sorted(
                [f for f in os.listdir(root_dir) if '_pose' in f and
                os.path.isfile(os.path.join(root_dir, f))
                and 'video{}'.format(self.video_selected) in f])

        self.frame_files = sorted([f for f in os.listdir(root_dir) if '_pose' in f and os.path.isfile(os.path.join(root_dir, f))])
        
        self.no_pose_repetitions = 0

        # Define the transformations
        self.transformS = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize spectrogram to correct shape for Mel-spectrogram
            transforms.RandomApply([AddGaussianNoise(0., 2)], p=0.3),
            transforms.RandomApply([FrequencyMasking(max_mask_size=8)], p=0.2),
        ])

        self.transformF = transforms.Compose([
            transforms.RandomApply([transforms.RandomCrop(240)], p=0.1),
            transforms.Resize((256, 256)),  # Resize frame to 256x256
            transforms.RandomApply([AddGaussianNoise(0., 50)], p=0.5),
            #transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.5)
        ])
        self.transformF2 = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize frame to 256x256
            #transforms.RandomApply([AddGaussianNoise(0., 10)], p=0.8),
            #transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.5)
        ])

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        if (idx >= len(self.frame_files)):
            return self.__getitem__(idx - 1)
        frame_file = self.frame_files[idx]
        frame_path = os.path.join(self.root_dir, frame_file)

        frame = np.load(frame_path)
        frame, is_pose = centralize_pose(frame, 20)

        if (not is_pose):
            self.frame_files.pop(idx)
            if (idx >= len(self.frame_files)):
                return self.__getitem__(idx)

        spectrogram_file = frame_file.replace('_pose', '_spectrogram')
        spectrogram_path = os.path.join(self.root_dir, spectrogram_file)
        spectrogram = np.load(spectrogram_path)

        # Get the previous frame
        if idx > 0:
            prev_frame_file = self.frame_files[idx - 1]
            prev_frame_path = os.path.join(self.root_dir, prev_frame_file)
            prev_frame = np.load(prev_frame_path)
            prev_frame, _ = centralize_pose(prev_frame, 20)
        else:
            prev_frame = np.zeros_like(frame)  # Use a zero array if there is no previous frame

        # Convert numpy arrays to PyTorch tensors with the same data type
        frame = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1)
        prev_frame = torch.from_numpy(prev_frame.astype(np.float32)).permute(2, 0, 1)
        spectrogram = torch.from_numpy(spectrogram.astype(np.float32)).unsqueeze(0)  # Add channel dimension

        # Apply transformations
        frame = self.transformF2(frame)
        prev_frame = self.transformF(prev_frame)
        spectrogram = self.transformS(spectrogram)

        # Normalize to [-1, 1]
        frame = (frame / 127.5) - 1
        prev_frame = (prev_frame / 127.5) - 1
        spectrogram = (spectrogram / 127.5) - 1

        return frame, prev_frame, spectrogram