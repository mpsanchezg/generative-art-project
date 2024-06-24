

import random
from src.load_data import load_data
from src.custom_tranformations import FrameSpectrogramDataset
from GAN import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Function to denormalize images
def denormalize(tensor):
    tensor = tensor * 0.5 + 0.5  # Denormalize from [-1, 1] to [0, 1]
    return tensor


# Function to check for NaNs in a tensor
def check_nan(tensor, name=""):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False


def plot_first_10_pairs_of_data(first_10_pairs):
    # Plot the first 10 pairs of data
    fig, axs = plt.subplots(10, 2, figsize=(10, 20))
    for i, (frame, prev_frame, spectrogram) in enumerate(first_10_pairs):
        frame = denormalize(frame.permute(1, 2, 0).numpy())  # Denormalize and rearrange dimensions
        spectrogram = denormalize(spectrogram.permute(1, 2, 0).numpy())  # Denormalize

        axs[i, 0].imshow(frame)
        axs[i, 1].imshow(spectrogram, cmap='viridis')
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')

    plt.show()


def plot_last_10_pairs_of_data(last_10_pairs):
    # Plot the last 10 pairs of data
    fig, axs = plt.subplots(10, 2, figsize=(10, 20))
    for i, (frame, prev_frame, spectrogram) in enumerate(last_10_pairs):
        frame = denormalize(frame.permute(1, 2, 0).numpy())  # Denormalize and rearrange dimensions
        spectrogram = denormalize(spectrogram.permute(1, 2, 0).numpy())  # Denormalize

        axs[i, 0].imshow(frame)
        axs[i, 1].imshow(spectrogram, cmap='viridis')
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')

    plt.show()


def plot_losses(losses_G, losses_D, val_iteration_steps, val_losses_G):
    # Plot the losses

    plt.figure(figsize=(10, 5))
    plt.plot(losses_G, label='Generator Loss')
    plt.plot(losses_D, label='Discriminator Loss')
    plt.plot(val_iteration_steps, val_losses_G, label='Validation Generator Loss', marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Generator, Discriminator, and Validation Generator Losses')
    plt.legend()
    plt.ylim(0, 20)  # Set the y-axis limits
    plt.grid(True)
    plt.show()


def train():

    # Hyperparameters
    hparams = {
        'batch_size': 16,
        'num_epochs': 10,
        'learning_rate': 0.00005,
        'betas': (0.5, 0.999),
        'num_val_samples': 4,
        'input_channels': 4,
        'output_channels': 3,
        'use_generated_frames_prob': 1
    }

    load_data()

    # Initialize dataset and dataloaders
    path = '/home/eduardo/projects/generative-art-project/data/raining/frames/'

    dataset = FrameSpectrogramDataset(path)
    dataset_size = len(dataset)
    split_index = int(0.9 * dataset_size)
    train_indices = list(range(split_index))
    val_indices = list(range(split_index, dataset_size))
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)  # Assuming hparams['batch_size'] is 16
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # TODO fix path


    print(len(dataset))

    # Get the first 10 pairs of data from the dataset
    first_10_pairs = [dataset[i] for i in range(10)]

    # Get the last 10 pairs of data from the dataset
    last_10_pairs = [dataset[i] for i in range(len(dataset) - 10, len(dataset))]
    # plot_first_10_pairs_of_data(first_10_pairs)
    # plot_last_10_pairs_of_data(last_10_pairs)

    # Initialize models, optimizers, and criteria
    netG = UNetGenerator().to(device)
    netD = MultiScaleDiscriminator(input_channels=hparams['input_channels'] + hparams['output_channels']).to(device)

    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)

    optimizerG = optim.Adam(netG.parameters(), lr=hparams['learning_rate'], betas=hparams['betas'])
    optimizerD = optim.Adam(netD.parameters(), lr=hparams['learning_rate'], betas=hparams['betas'])

    schedulerG = StepLR(optimizerG, step_size=10, gamma=0.1)
    schedulerD = StepLR(optimizerD, step_size=10, gamma=0.1)

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

    for epoch in range(hparams['num_epochs']):
        for i, (real_frames, prev_frames, spectrograms) in enumerate(dataloader):
            real_frames = real_frames.to(device)
            prev_frames = prev_frames.to(device)
            spectrograms = spectrograms.to(device)

            hidden_state = None
            inputs = torch.cat((spectrograms, prev_frames), dim=1)  # (N, 4, H, W)

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
            real_labels = [torch.ones_like(out).to(device) for out in output_real]
            losses_D_real = [criterion_gan(out, label) for out, label in zip(output_real, real_labels)]
            lossD_real = sum(losses_D_real)

            fake_inputs = torch.cat((inputs, fake_frames.detach()), 1)  # (N, 7, H, W)
            output_fake = netD(fake_inputs)
            fake_labels = [torch.zeros_like(out_fake).to(device) for out_fake in output_fake]
            lossD_fake = sum(
                [criterion_gan(out_fake, fake_label) for out_fake, fake_label in zip(output_fake, fake_labels)])

            gradient_penalty = compute_gradient_penalty(netD, real_inputs.data, fake_inputs.data)

            # Check for NaNs before the backward pass
            if any([check_nan(tensor, name) for tensor, name in
                    zip([lossD_real, lossD_fake, gradient_penalty], ['lossD_real', 'lossD_fake', 'gradient_penalty'])]):
                continue

            lossD = (lossD_real + lossD_fake) * 0.5 + gradient_penalty
            lossD.backward()
            torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
            optimizerD.step()

            optimizerG.zero_grad()
            # Compute losses and gradients for Generator G
            fake_inputs = torch.cat((inputs, fake_frames), 1)  # (N, 7, H, W)
            output_fake = netD(fake_inputs)
            loss_gan = sum([criterion_gan(out_fake, real_label) for out_fake, real_label in zip(output_fake, real_labels)])
            loss_l1 = criterion_l1(fake_frames, real_frames)
            loss_perceptual = criterion_perceptual(fake_frames, real_frames)
            loss_feature_matching = feature_matching_loss(real_inputs, fake_inputs)

            # Check for NaNs before the backward pass
            if any([check_nan(tensor, name) for tensor, name in
                    zip([loss_gan, loss_l1, loss_perceptual, loss_feature_matching],
                        ['loss_gan', 'loss_l1', 'loss_perceptual', 'loss_feature_matching'])]):
                continue

            lossG = (loss_gan * 5 + loss_l1 * 50 + loss_perceptual * 10 + loss_feature_matching * 10)
            lossG.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            optimizerG.step()

            lossG = lossG.to(device)
            lossD = lossD.to(device)
            losses_G.append(lossG.item())
            losses_D.append(lossD.item())

            if i % 400 == 0:
                print(f'Epoch [{epoch}/{hparams["num_epochs"]}], Step [{i}/{len(dataloader)}], '
                      f'Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}')
                with torch.no_grad():
                    val_lossG = 0.0
                    for val_real_frames, val_prev_frames, val_spectrograms in val_loader:
                        val_real_frames = val_real_frames.to(device)
                        val_prev_frames = val_prev_frames.to(device)
                        val_spectrograms = val_spectrograms.to(device)

                        val_inputs = torch.cat((val_spectrograms, val_prev_frames), dim=1)  # (N, 4, H, W)
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
                        if any([check_nan(tensor, name) for tensor, name in
                                zip([val_loss_gan, val_loss_l1, val_loss_perceptual, val_loss_feature_matching],
                                    ['val_loss_gan', 'val_loss_l1', 'val_loss_perceptual', 'val_loss_feature_matching'])]):
                            continue

                        val_lossG += (
                                    val_loss_gan * 5 + val_loss_l1 * 50 + val_loss_perceptual * 10 + val_loss_feature_matching * 10).item()

                val_lossG /= len(val_loader)
                val_losses_G.append(val_lossG)
                val_iteration_steps.append(epoch * len(dataloader) + i)
                print(f'Validation Loss G: {val_lossG:.4f}')

        schedulerG.step()
        schedulerD.step()

        # Visualize input and output images
        with torch.no_grad():
            random_batch = next(iter(dataloader))
            prev_frames_batch = random_batch[1][:hparams['num_val_samples']].to(device)
            spectrograms_batch = random_batch[2][:hparams['num_val_samples']].to(device)
            fake_frames_batch, _ = netG(torch.cat((spectrograms_batch, prev_frames_batch), dim=1))

            prev_frames_denorm = denormalize(prev_frames_batch)
            fake_frames_denorm = denormalize(fake_frames_batch)

            spect_grid = make_grid(spectrograms_batch, nrow=hparams['num_val_samples'])
            prev_img_grid = make_grid(prev_frames_denorm, nrow=hparams['num_val_samples'])
            fake_img_grid = make_grid(fake_frames_denorm, nrow=hparams['num_val_samples'])

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title('Spectrogram')
            plt.imshow(spect_grid.permute(1, 2, 0).cpu().numpy())
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


if __name__ == "__main__":
    train()

