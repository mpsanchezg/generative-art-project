import matplotlib.pyplot as plt
import torch


def denormalize(tensor):
    """"
    Ensure consistent normalization/de-normalization functions
    """
    tensor = tensor * 0.5 + 0.5  # Denormalize from [-1, 1] to [0, 1]
    return tensor


# Function to check for NaNs in a tensor
def check_nan(tensor, name=""):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

# Function to generate smoothed labels
def get_smoothed_labels(labels, device, smooth_real=True):
    if smooth_real:
        return torch.FloatTensor(labels.size()).uniform_(0.7, 1.2).to(device)
    else:
        return torch.FloatTensor(labels.size()).uniform_(0.0, 0.3).to(device)


def plot_first_10_pairs_of_data(first_10_pairs, logger=None):
    # Plot the first 10 pairs of data
    fig, axs = plt.subplots(10, 3, figsize=(15, 20))
    for i, (frame, prev_frame, spectrogram) in enumerate(first_10_pairs):
        frame = denormalize(frame.permute(1, 2, 0).numpy())  # Denormalize and rearrange dimensions
        prev_frame = denormalize(prev_frame.permute(1, 2, 0).numpy())  # Denormalize and rearrange dimensions
        spectrogram = denormalize(spectrogram.permute(1, 2, 0).numpy())  # Denormalize

        axs[i, 0].imshow(prev_frame)
        axs[i, 1].imshow(frame)
        axs[i, 2].imshow(spectrogram, cmap='viridis')
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')
        axs[i, 2].axis('off')

    if logger:
        logger.log_plots(plot=plt, title="first_10_pairs")
    else:
        plt.show()


def plot_last_10_pairs_of_data(last_10_pairs, logger=None):
    # Plot the last 10 pairs of data
    fig, axs = plt.subplots(10, 3, figsize=(15, 20))
    for i, (frame, prev_frame, spectrogram) in enumerate(last_10_pairs):
        frame = denormalize(frame.permute(1, 2, 0).numpy())  # Denormalize and rearrange dimensions
        prev_frame = denormalize(prev_frame.permute(1, 2, 0).numpy())  # Denormalize and rearrange dimensions
        spectrogram = denormalize(spectrogram.permute(1, 2, 0).numpy())  # Denormalize

        axs[i, 0].imshow(prev_frame)
        axs[i, 1].imshow(frame)
        axs[i, 2].imshow(spectrogram, cmap='viridis')
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')
        axs[i, 2].axis('off')

    if logger:
        logger.log_plots(plot=plt, title="last_10_pairs")
    else:
        plt.show()


def plot_losses(losses_G, losses_D, val_iteration_steps, val_losses_G, logger=None):
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

    if logger:
        logger.log({"plot_losses": plt})
    else:
        plt.show()

