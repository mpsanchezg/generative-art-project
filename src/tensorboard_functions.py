import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from datetime import datetime
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

    def log_training_loss(
            self,
            step: int,
            lossG: torch.Tensor,
            lossD: torch.Tensor,
            label: str,
    ):

        self.writer.add_scalar('Train_loss_{}/loss_g'.format(label), lossG, step)
        self.writer.add_scalar('Train_loss_{}/loss_d'.format(label), lossD, step)

    def log_training_loss_detailed(
            self,
            step: int,
            lossD,
            lossD_real,
            lossD_fake,
            gradient_penalty,
            lossG,
            loss_gan,
            loss_l1,
            loss_perceptual,
            loss_feature_matching,
    ):
        self.writer.add_scalar('Train_loss_detailed/lossD', lossD, step)
        self.writer.add_scalar('Train_loss_detailed/lossD_real', lossD_real, step)
        self.writer.add_scalar('Train_loss_detailed/lossD_fake', lossD_fake, step)
        self.writer.add_scalar('Train_loss_detailed/gradient_penalty', gradient_penalty, step)

        self.writer.add_scalar('Train_loss_detailed/lossG', lossG, step)
        self.writer.add_scalar('Train_loss_detailed/loss_gan', loss_gan, step)
        self.writer.add_scalar('Train_loss_detailed/loss_l1', loss_l1, step)
        self.writer.add_scalar('Train_loss_detailed/loss_perceptual', loss_perceptual, step)
        self.writer.add_scalar('Train_loss_detailed/loss_feature_matching', loss_feature_matching, step)

    def log_validation_loss(
            self,
            step: int,
            val_loss_gan: float,
            val_loss_l1: torch.Tensor,
            val_loss_perceptual: torch.Tensor,
            val_loss_feature_matching: torch.Tensor,
            val_lossG: float,
    ):
        self.writer.add_scalar('Validation_epoch/loss_gan', val_loss_gan, step)
        self.writer.add_scalar('Validation_epoch/loss_l1', val_loss_l1, step)
        self.writer.add_scalar('Validation_epoch/loss_perceptual', val_loss_perceptual, step)
        self.writer.add_scalar('Validation_epoch/loss_feature_matching', val_loss_feature_matching, step)
        self.writer.add_scalar('Validation_epoch/loss_g', val_lossG, step)

        # Logging the weights values and grads histograms.
        # for name, weight in model.encoder.named_parameters():
        #    self.writer.add_histogram(f"{name}/value", weight)
        #    self.writer.add_histogram(f"{name}/grad", weight.grad)
        # pass


