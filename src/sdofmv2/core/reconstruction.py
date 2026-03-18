# Adapted from: FDL 2021 Solar Drag - Feature Extraction
# Learning the solar latent space: sigma-variational autoencoders for multiple channel solar imaging
# https://ml4physicalsciences.github.io/2021/files/NeurIPS_ML4PS_2021_83.pdf
from functools import lru_cache

import numpy as np
import torch

RADIUS_FRACTION_OF_IMAGE = 0.40625
METRICS_METHODS = {
    "flux_difference": lambda real, generated: relative_total_flux_error(
        real, generated
    ),
    "ppe10s": lambda real, generated: pixel_percentage_error(real, generated, 0.1),
    "ppe50s": lambda real, generated: pixel_percentage_error(real, generated, 0.5),
    "rms_contrast_measure": lambda real, generated: rms_contrast_measure(
        real, generated
    ),
    "pixel_correlation": lambda real, generated: pixel_correlation_coefficient(
        real, generated
    ),
    "rmse_intensity": lambda real, generated: rmse_intensity(real, generated),
}

METRICS = list(METRICS_METHODS.keys())
MIN_CLIP = 1e-2
MAX_CLIP = np.inf
MASK_DISK = True


def get_metrics_for_masked_patches(real, generated, channels):
    """Calculates evaluation metrics for flattened image patches across channels.

    Args:
        real (np.ndarray): Flattened array of real image data.
        generated (np.ndarray): Flattened array of generated image data.
        channels (list[str]): List of names for the image channels.

    Returns:
        dict: A nested dictionary where keys are channel names and values are
            dictionaries mapping metric names to lists of calculated values.
    """
    metrics = {}
    for channel in channels:
        metrics[channel] = {}
        for metric in METRICS:
            metrics[channel][metric] = []

    for c, channel in enumerate(channels):
        real_channel = real[c].flatten()
        generated_channel = generated[c].flatten()

        for metric, function in METRICS_METHODS.items():
            metrics[channel][metric].append(function(real_channel, generated_channel))

    return metrics


def get_metrics(real, generated, channels, mask_disk=MASK_DISK):
    """Computes metrics for single images with optional circular masking.

    Expects input in CxHxW format. Applies a disk mask to the first three
    channels if mask_disk is enabled.

    Args:
        real (torch.Tensor | np.ndarray): The ground truth image data.
        generated (torch.Tensor | np.ndarray): The predicted or generated image data.
        channels (list[str]): List of names for the image channels.
        mask_disk (bool): Whether to apply a circular mask to the first three channels.

    Returns:
        dict: A nested dictionary containing calculated metrics for each channel.
    """
    ## Expect CxHxW
    if torch.is_tensor(real):
        real = real.cpu().detach().numpy()
        generated = generated.cpu().detach().numpy()

    _, _, image_size = real.shape
    mask = disk_mask(image_size)

    metrics = {}
    for channel in channels:
        metrics[channel] = {}
        for metric in METRICS:
            metrics[channel][metric] = []

    for c, channel in enumerate(channels):
        if mask_disk and c <= 2:
            real_channel = (real[c, :, :] * mask).flatten()
            generated_channel = (generated[c, :, :] * mask).flatten()
        else:
            real_channel = real[c, :, :].flatten()
            generated_channel = generated[c, :, :].flatten()

        for metric, function in METRICS_METHODS.items():
            metrics[channel][metric].append(function(real_channel, generated_channel))

    return metrics


def get_batch_metrics(real_batch, generated_batch, channels):
    """Calculates the average metrics across a batch of images.

    Args:
        real_batch (np.ndarray): Batch of real images in BxCxHxW format.
        generated_batch (np.ndarray): Batch of generated images in BxCxHxW format.
        channels (list[str]): List of names for the image channels.

    Returns:
        dict: A dictionary containing the mean value for each metric per channel.
    """
    ## Expect BxCxHxW
    batch_size = real_batch.shape[0]
    metrics_list = []
    for batch_idx in range(batch_size):
        sample_metrics = get_metrics(
            real_batch[batch_idx, :, :, :],
            generated_batch[batch_idx, :, :, :],
            channels,
        )
        metrics_list.append(sample_metrics)

    # print("final state is: ", len(metrics_list))
    merged_metrics = merge_metrics(metrics_list)
    return mean_metrics(merged_metrics)


@lru_cache(maxsize=10)
def disk_mask(image_size):
    """Generates a circular binary mask centered in a square array.

    Args:
        image_size (int): The height and width of the square mask.

    Returns:
        np.ndarray: A 2D array of shape (image_size, image_size) where pixels
            inside the circle are 1 and pixels outside are 0.
    """
    img_half = image_size / 2
    radius = int(RADIUS_FRACTION_OF_IMAGE * float(image_size))
    disk_mask = np.zeros((image_size, image_size))
    for h in range(image_size):
        for w in range(image_size):
            distance_to_centre = np.sqrt((h - img_half) ** 2 + (w - img_half) ** 2)
            if distance_to_centre < radius:
                disk_mask[h][w] = 1
    return disk_mask


def merge_metrics(metrics_list):
    """Aggregates a list of metric dictionaries into a single dictionary.

    Args:
        metrics_list (list[dict]): A list of dictionaries, where each dictionary
            contains metrics for a single sample.

    Returns:
        dict: A dictionary where each metric for each channel contains a list
            of values from all samples in the input list.
    """
    # print(metrics_list)
    channels = list(metrics_list[0].keys())
    merged_metrics = {}
    for channel in channels:
        merged_metrics[channel] = {}
        for metric in METRICS:
            merged_metrics[channel][metric] = []
            for cm in metrics_list:
                # print(cm[channel][metric])
                merged_metrics[channel][metric] += cm[channel][metric]
                # print(merged_metrics[channel][metric])

    return merged_metrics


def mean_metrics(metrics):
    """Computes the mean for each metric in a merged metrics dictionary.

    Args:
        metrics (dict): A dictionary where values are lists of metric results.

    Returns:
        dict: A dictionary containing the arithmetic mean of each metric list.
    """
    mean_metrics = {}
    for channel, channel_metrics in metrics.items():
        mean_metrics[channel] = {}
        for metric in METRICS:
            mean_metrics[channel][metric] = np.mean(channel_metrics[metric])

    return mean_metrics


def pixel_percentage_error(real, generated, threshold_ptc):
    """Calculates the fraction of pixels with a relative error below a threshold.

    Args:
        real (np.ndarray): Array of ground truth pixel values.
        generated (np.ndarray): Array of predicted pixel values.
        threshold_ptc (float): The relative error threshold (e.g., 0.1 for 10%).

    Returns:
        float: The mean pixel percentage error across the arrays.
    """
    difference = real - generated
    fraction = np.divide(difference, real, where=real != 0)
    absolute_fraction = np.abs(fraction)
    ppe_binary = (absolute_fraction < threshold_ptc).astype(int)
    mean_ppe = ppe_binary.mean()

    return mean_ppe


def pixel_correlation_coefficient(real, generated):
    """Computes the Pearson correlation coefficient between two arrays.

    Args:
        real (np.ndarray): Array of ground truth pixel values.
        generated (np.ndarray): Array of predicted pixel values.

    Returns:
        float: The correlation coefficient between the two inputs.
    """
    correlation = np.corrcoef(real, generated)[0, 1]
    return correlation


def rms_contrast_measure(real, generated):
    """Calculates the RMS difference between the real mean and generated pixels.

    Args:
        real (np.ndarray): Array of ground truth pixel values.
        generated (np.ndarray): Array of predicted pixel values.

    Returns:
        float: The root mean square of the difference between the real mean
            and the generated values.
    """
    difference_squared = np.power(real.mean() - generated, 2)
    mean_difference = difference_squared.mean()
    rms = np.sqrt(mean_difference)
    return rms


def relative_total_flux_error(real, generated):
    """Calculates the relative difference in total flux (sum of pixel values).

    Args:
        real (np.ndarray): Array of ground truth pixel values.
        generated (np.ndarray): Array of predicted pixel values.

    Returns:
        float: The relative error of the total sum of generated pixels
            compared to real pixels.
    """
    return (generated.sum() - real.sum()) / real.sum()


def rmse_intensity(real, generated):
    """Calculates the root mean square error (RMSE) between two arrays.

    Args:
        real (np.ndarray): Array of ground truth pixel values.
        generated (np.ndarray): Array of predicted pixel values.

    Returns:
        float: The root mean square error of the pixel intensities.
    """
    difference_squared = np.power(real - generated, 2)
    mean_difference = difference_squared.mean()
    rms = np.sqrt(mean_difference)
    return rms
