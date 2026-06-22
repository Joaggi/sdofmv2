# Adapted from: FDL 2021 Solar Drag - Feature Extraction
# Learning the solar latent space: sigma-variational autoencoders for multiple channel solar imaging
# https://ml4physicalsciences.github.io/2021/files/NeurIPS_ML4PS_2021_83.pdf

import torch

RADIUS_FRACTION_OF_IMAGE = 0.40625


def compute_metrics_pytorch(
    real: torch.Tensor, generated: torch.Tensor, mask: torch.Tensor, channels: list[str]
) -> dict:
    """Computes all reconstruction metrics on the GPU using vectorized operations."""

    # real, gen: [B, C, T, H, W]
    # mask: [B, T, H, W] (True for patches/pixels to evaluate)

    # Expand mask to all channels [B, C, T, H, W]
    mask_c = mask.unsqueeze(1).expand(-1, real.shape[1], -1, -1, -1)

    # Select masked pixels
    # Using masked_select results in [N_masked_total] which doesn't keep channel info.
    # We need to keep channels, so let's use view + boolean indexing
    # Mask [B, C, T, H, W] -> [B*C*T*H*W]
    # This is trickier than I thought for channel-wise results.
    # Let's use mean over spatial-temporal dimensions

    # [B, C, T, H, W]
    mask_c = mask_c.float()

    # Weighted mean/sum for metrics
    # RMSE: [C]
    diff = real - generated
    mse = (diff**2) * mask_c
    rmse = torch.sqrt(mse.sum(dim=[0, 2, 3, 4]) / (mask_c.sum(dim=[0, 2, 3, 4]) + 1e-6))

    mask_sum = mask_c.sum(dim=[0, 2, 3, 4])
    mse_mean = mse.sum(dim=[0, 2, 3, 4]) / (mask_sum + 1e-6)
    mae = (torch.abs(diff) * mask_c).sum(dim=[0, 2, 3, 4]) / (mask_sum + 1e-6)

    # Flux Error: [C]
    real_masked = real * mask_c
    gen_masked = generated * mask_c
    real_sum = real_masked.sum(dim=[0, 2, 3, 4])
    gen_sum = gen_masked.sum(dim=[0, 2, 3, 4])
    flux_error = (gen_sum - real_sum) / (real_sum + 1e-6)

    # PPE: [C]
    ppe10 = ((torch.abs(diff / (real + 1e-6)) < 0.1).float() * mask_c).sum(dim=[0, 2, 3, 4]) / (
        mask_c.sum(dim=[0, 2, 3, 4]) + 1e-6
    )
    ppe50 = ((torch.abs(diff / (real + 1e-6)) < 0.5).float() * mask_c).sum(dim=[0, 2, 3, 4]) / (
        mask_c.sum(dim=[0, 2, 3, 4]) + 1e-6
    )

    # R2 and Correlation: [C]
    # Calculate masked means
    real_mean = real_sum / (mask_c.sum(dim=[0, 2, 3, 4]) + 1e-6)

    # Need to broadcast real_mean for variance calculation
    # [B, C, T, H, W]
    real_mean_expanded = real_mean.view(1, -1, 1, 1, 1)

    # SS_tot: [C]
    ss_tot = (((real - real_mean_expanded) ** 2) * mask_c).sum(dim=[0, 2, 3, 4])

    # SS_res: [C]
    ss_res = mse.sum(dim=[0, 2, 3, 4])

    # R2: [C]
    r2 = 1 - (ss_res / (ss_tot + 1e-6))

    # Pearson Correlation: [C]
    gen_mean = gen_sum / (mask_c.sum(dim=[0, 2, 3, 4]) + 1e-6)
    gen_mean_expanded = gen_mean.view(1, -1, 1, 1, 1)

    cov = (((real - real_mean_expanded) * (generated - gen_mean_expanded)) * mask_c).sum(
        dim=[0, 2, 3, 4]
    )
    ss_gen = (((generated - gen_mean_expanded) ** 2) * mask_c).sum(dim=[0, 2, 3, 4])

    correlation = cov / (torch.sqrt(ss_tot + 1e-6) * torch.sqrt(ss_gen + 1e-6) + 1e-6)

    metrics = {}
    for c, channel in enumerate(channels):
        metrics[channel] = {
            "mse": mse_mean[c].item(),
            "mae": mae[c].item(),
            "rmse_intensity": rmse[c].item(),
            "flux_difference": flux_error[c].item(),
            "ppe10s": ppe10[c].item(),
            "ppe50s": ppe50[c].item(),
            "r2_score": r2[c].item(),
            "pixel_correlation": correlation[c].item(),
        }

    return metrics
