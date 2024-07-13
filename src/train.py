import os
import sys

# insert root directory into python module search path
sys.path.insert(1, os.getcwd())

import argparse
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid

from model import UNetGenerator, MultiScaleDiscriminator, weights_init_normal, PerceptualLoss, FeatureMatchingLoss, \
    compute_gradient_penalty
from src.config import DATA_DIR
from src.custom_tranformations import FrameSpectrogramDataset
from src.extract_frames import extract_frames
from utils import denormalize, check_nan, plot_losses, get_smoothed_labels, plot_last_10_pairs_of_data, plot_first_10_pairs_of_data
from tensorboard_functions import TensorboardLogger
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--video_selected", type=int, default=None, help="Select a specific video to train on")
parser.add_argument("--extract_frames", type=bool, default=False, help="Extract frames from videos")

args = parser.parse_args()


def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Hyperparameters
    hparams = {
        'batch_size': 25,
        'num_epochs': 20,
        'learning_rate': 0.00001,
        'betas': (0.5, 0.999),
        'num_val_samples': 4,
        'input_channels': 4,
        'output_channels': 3,
        'use_generated_frames_prob': 1
    }
    task = 'train'

    logger = TensorboardLogger(task)

    if args.extract_frames:
        extract_frames(args.number_of_videos)

    # Initialize dataset and dataloaders
    path = os.path.join(DATA_DIR, "frames")

    dataset = FrameSpectrogramDataset(path, video_selected=args.video_selected)

    dataset_size = len(dataset)
    split_index = int(0.9 * dataset_size)
    train_indices = list(range(split_index))
    val_indices = list(range(split_index, dataset_size))
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    dataloader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=False)  # Assuming hparams['batch_size'] is 16
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False)

    # Get the first 10 pairs of data from the dataset
    # first_10_pairs = [dataset[i] for i in range(10)]
    # plot_first_10_pairs_of_data(first_10_pairs)
    # plot_first_10_pairs_of_data(first_10_pairs, logger=logger)

    # Get the last 10 pairs of data from the dataset
    # last_10_pairs = [dataset[i] for i in range(len(dataset) - 10, len(dataset))]
    # plot_last_10_pairs_of_data(last_10_pairs)
    # plot_last_10_pairs_of_data(last_10_pairs, logger=logger)
    # logger.finish()

    # Initialize models, optimizers, and criteria
    netG = UNetGenerator().to(device)
    netD = MultiScaleDiscriminator(input_channels=hparams['input_channels'] + hparams['output_channels']).to(device)

    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)
    
    optimizerG = optim.Adam(netG.parameters(), lr=hparams['learning_rate'], betas=hparams['betas'])
    optimizerD = optim.Adam(netD.parameters(), lr=hparams['learning_rate'], betas=hparams['betas'])

    schedulerG = StepLR(optimizerG, step_size=1, gamma=0.8)
    schedulerD = StepLR(optimizerD, step_size=1, gamma=0.8)

    criterion_gan = nn.BCEWithLogitsLoss().to(device)
    criterion_l1 = nn.L1Loss()
    criterion_perceptual = PerceptualLoss().to(device)
    feature_matching_loss = FeatureMatchingLoss(netD).to(device)

    # Training loop
    netG.train()
    losses_G = []
    losses_D = []
    val_losses_G = []
    val_iteration_steps = []

    tensorboard_counter = 0
    for epoch in range(hparams['num_epochs']):
        lossG = np.NaN
        lossD = np.NaN
        
        for i, (real_frames, prev_frames, spectrograms) in enumerate(dataloader):
            tensorboard_counter += 1
            real_frames = real_frames.to(device)
            prev_frames = prev_frames.to(device)
            spectrograms = spectrograms.to(device)

            inputs = torch.cat((spectrograms, prev_frames), dim=1)  # (N, 4, H, W)

            batch_size = inputs.shape[0]  # Get batch size from input data
            if batch_size == hparams['batch_size']:
                hidden_state = netG.convlstm.init_hidden(hparams['batch_size'], (1,1))

                generated_frames = []
                fake_frames, hidden_state = netG(inputs, hidden_state)
                generated_frames.append(fake_frames)
                use_generated = random.random() < hparams['use_generated_frames_prob']
                if use_generated and len(generated_frames) > 1:
                    prev_frames = generated_frames[-1]

                inputs = torch.cat((spectrograms, prev_frames), dim=1)
                real_inputs = torch.cat((inputs, real_frames), 1)  # (N, 7, H, W)

                optimizerD.zero_grad()
                # Compute losses for Discriminator D
                output_real = netD(real_inputs)
                real_labels = [get_smoothed_labels(out, device, smooth_real=True) for out in output_real]
                losses_D_real = [criterion_gan(out, label) for out, label in zip(output_real, real_labels)]
                lossD_real = sum(losses_D_real)

                fake_inputs = torch.cat((inputs, fake_frames.detach()), 1)  # (N, 7, H, W)
                output_fake = netD(fake_inputs)
                fake_labels = [get_smoothed_labels(out_fake, device, smooth_real=False) for out_fake in output_fake]
                lossD_fake = sum([criterion_gan(out_fake, fake_label) for out_fake, fake_label in zip(output_fake, fake_labels)])

                gradient_penalty = compute_gradient_penalty(netD, real_inputs.data, fake_inputs.data)

                # Check for NaNs before the backward pass
                if any([check_nan(tensor, name) for tensor, name in zip([lossD_real, lossD_fake, gradient_penalty], ['lossD_real', 'lossD_fake', 'gradient_penalty'])]):
                    continue

                lossD = (lossD_real + lossD_fake) * 0.9 + gradient_penalty * 0.1
                lossD.backward()
                torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
                optimizerD.step()

                optimizerG.zero_grad()
                # Compute losses and gradients for Generator G
                fake_inputs = torch.cat((inputs, fake_frames), 1)  # (N, 7, H, W)
                output_fake = netD(fake_inputs)
                loss_gan = sum([criterion_gan(out_fake, get_smoothed_labels(out_fake, device, smooth_real=True)) for out_fake in output_fake])
                loss_l1 = criterion_l1(fake_frames, real_frames)
                loss_perceptual = criterion_perceptual(fake_frames, real_frames)
                loss_feature_matching = feature_matching_loss(real_inputs, fake_inputs)

                # Check for NaNs before the backward pass
                if any([check_nan(tensor, name) for tensor, name in zip([loss_gan, loss_l1, loss_perceptual, loss_feature_matching], ['loss_gan', 'loss_l1', 'loss_perceptual', 'loss_feature_matching'])]):
                    continue

                lossG = (loss_gan * 2 + loss_l1 * 1 + loss_perceptual * 1 + loss_feature_matching * 1)
                lossG.backward()
                torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
                optimizerG.step()

                lossG = lossG.to(device)
                lossD = lossD.to(device)
                losses_G.append(lossG.item())
                losses_D.append(lossD.item())

            if i % 500 == 0:
                print(f'Epoch [{epoch+1}/{hparams["num_epochs"]}], Step [{i}/{len(dataloader)}], '
                    f'Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}')

                with torch.no_grad():
                    val_lossG = 0.0
                    for val_real_frames, val_prev_frames, val_spectrograms in val_loader:
                        val_real_frames = val_real_frames.to(device)
                        val_prev_frames = val_prev_frames.to(device)
                        val_spectrograms = val_spectrograms.to(device)

                        val_inputs = torch.cat((val_spectrograms, val_prev_frames), dim=1)  # (N, 4, H, W)

                            
                        if val_inputs.shape[0] == hparams['batch_size']:
                        
                            val_fake_frames, _ = netG(val_inputs, hidden_state)

                            val_fake_inputs = torch.cat((val_inputs, val_fake_frames), 1)  # (N, 7, H, W)

                            val_output_fake = netD(val_fake_inputs)
                            real_labels = [torch.ones_like(out).to(device) for out in val_output_fake]

                            losses_val_gan = [criterion_gan(out, label) for out, label in zip(val_output_fake, real_labels)]
                            val_loss_gan = sum(losses_val_gan)
                            val_loss_l1 = criterion_l1(val_fake_frames, val_real_frames)
                            val_loss_perceptual = criterion_perceptual(val_fake_frames, val_real_frames)
                            val_loss_feature_matching = feature_matching_loss(real_inputs, val_fake_inputs)

                            # Check for NaNs before summing the validation loss
                            if any([check_nan(tensor, name) for tensor, name in zip([val_loss_gan, val_loss_l1, val_loss_perceptual, val_loss_feature_matching], ['val_loss_gan', 'val_loss_l1', 'val_loss_perceptual', 'val_loss_feature_matching'])]):
                                continue


                            val_lossG += ((val_loss_gan * 2 + val_loss_l1 * 1 + val_loss_perceptual * 1 + val_loss_feature_matching * 1)).item()

                            logger.log_validation_loss(
                                tensorboard_counter,
                                val_loss_gan,
                                val_loss_l1,
                                val_loss_perceptual,
                                val_loss_feature_matching,
                                val_lossG,
                            )

                val_lossG /= len(val_loader)
                val_losses_G.append(val_lossG)
                val_iteration_steps.append(epoch * len(dataloader) + i)
                print(f'Validation Loss G: {val_lossG:.4f}')


            logger.log_training_loss_detailed(
                tensorboard_counter,
                lossD,
                lossD_real,
                lossD_fake,
                gradient_penalty,
                lossG,
                loss_gan,
                loss_l1,
                loss_perceptual,
                loss_feature_matching
            )

            logger.log_training_loss(
                epoch,
                lossG,
                lossD,
                label='simple'
            )

        logger.log_training_loss(
            epoch,
            lossG,
            lossD,
            label='epoch'
        )
        schedulerG.step()
        schedulerD.step()

        # Visualize input and output images
        with torch.no_grad():
            # Using the last batch of the epoch
            prev_frames_batch = prev_frames[:hparams['num_val_samples']]
            spectrograms_batch = spectrograms[:hparams['num_val_samples']]
            fake_frames_batch = fake_frames[:hparams['num_val_samples']]

            prev_frames_denorm = denormalize(prev_frames_batch)
            fake_frames_denorm = denormalize(fake_frames_batch)

            spectrograms_batch = spectrograms_batch.squeeze(1)  # Remove the channel dimension for single-channel spectrograms

            spect_grid = make_grid(spectrograms_batch.unsqueeze(1), nrow=hparams['num_val_samples'], normalize=True)
            prev_img_grid = make_grid(prev_frames_denorm, nrow=hparams['num_val_samples'])
            fake_img_grid = make_grid(fake_frames_denorm, nrow=hparams['num_val_samples'])

            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.title('Spectrogram')
            plt.imshow(spect_grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title('Previous Frames')
            plt.imshow(prev_img_grid.permute(1, 2, 0).cpu().numpy())
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title('Generated Frames')
            plt.imshow(fake_img_grid.permute(1, 2, 0).cpu().numpy())
            plt.axis('off')

            plt.show()

        plot_losses(losses_G, losses_D, val_iteration_steps, val_losses_G)

    # Save the model weights
    torch.save(netG.state_dict(), "model_weights_V8.pth")


if __name__ == "__main__":
    train()

