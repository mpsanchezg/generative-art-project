import numpy as np

import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt

from datetime import datetime
from typing import Optional

from src.config import PROJECT_NAME


class WandbLogger:

    def __init__(
        self,
        task: str,
        hparams: dict,
        # model: nn.Module,
    ):
        wandb.login()
        wandb.init(project=PROJECT_NAME, config=hparams)
        wandb.run.name = f'{task}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        # start a new wandb run to track this script
        # TODO: Log weights and gradients to wandb. Doc: https://docs.wandb.ai/ref/python/watch

    @staticmethod
    def log_plots(
            plot: plt.Figure,
            title: str,
    ):
        # Save the plot as an image
        plt.savefig(f'{title}.png')

        # Log the image to W&B
        wandb.log({title: wandb.Image(f'{title}.png')})

    @staticmethod
    def finish():
        wandb.finish()

    def log_reconstruction_training(
        self,
        model: nn.Module,
        epoch: int,
        train_loss_avg: np.ndarray,
        val_loss_avg: np.ndarray,
        reconstruction_grid: Optional[torch.Tensor] = None,
    ):

        # TODO: Log train reconstruction loss to wandb

        # TODO: Log validation reconstruction loss to wandb

        # TODO: Log a batch of reconstructed images from the validation set

        pass


    def log_classification_training(
        self,
        epoch: int,
        train_loss_avg: float,
        val_loss_avg: float,
        train_acc_avg: float,
        val_acc_avg: float,
        fig: plt.Figure,
    ):
        # TODO: Log confusion matrix figure to wandb


        # TODO: Log validation loss to wandb
        #  Tip: use the tag 'Classification/val_loss'


        # TODO: Log validation accuracy to wandb
        #  Tip: use the tag 'Classification/val_acc'


        # TODO: Log training loss to wandb
        #  Tip: use the tag 'Classification/train_loss'


        # TODO: Log train accuracy to wandb
        #  Tip: use the tag 'Classification/train_acc'


        pass

    @staticmethod
    def log_plots(
            plot: plt.Figure,
            title: str,
    ):
        # Log the plots to wandb
        wandb.log({title: plot})

