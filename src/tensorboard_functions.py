import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from typing import Optional
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:

    def __init__(
        self,
        task: str,
    ):
        # Define the folder where we will store all the tensorboard logs
        logdir = os.path.join("logs", f"{task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

        # TODO: Initialize Tensorboard Writer with the previous folder 'logdir'
        self.writer = SummaryWriter(log_dir=logdir)

    def log_plots(
            self,
            plot: plt.Figure,
            title: str,
    ):
        # Save the plot as an image
        plt.savefig(f'{title}.png')

        # Log the image to W&B
        self.writer.add_image('Reconstruction/reconstruction_grid', reconstruction_grid, epoch)
        self.writer.log({title: wandb.Image(f'{title}.png')})

    def log_reconstruction_training(
        self,
        model: nn.Module,
        epoch: int,
        train_loss_avg: np.ndarray,
        val_loss_avg: np.ndarray,
        reconstruction_grid: Optional[torch.Tensor] = None,
    ):

        # TODO: Log train reconstruction loss to tensorboard.
        #  Tip: use "Reconstruction/train_loss" as tag
        self.writer.add_scalar('Reconstruction/train_loss', train_loss_avg, epoch)

        # TODO: Log validation reconstruction loss to tensorboard.
        #  Tip: use "Reconstruction/val_loss" as tag
        self.writer.add_scalar('Reconstruction/val_loss', val_loss_avg, epoch)

        # TODO: Log a batch of reconstructed images from the validation set.
        #  Use the reconstruction_grid variable returned above.
        self.writer.add_image('Reconstruction/reconstruction_grid', reconstruction_grid, epoch)

        # TODO: Log the weights values and grads histograms.
        #  Tip: use f"{name}/value" and f"{name}/grad" as tags
        for name, weight in model.encoder.named_parameters():
            self.writer.add_histogram(f"{name}/value", weight)
            self.writer.add_histogram(f"{name}/grad", weight.grad)
            # continue # remove this line when you complete the code
        pass


