import argparse
from pathlib import Path

import pandas as pd
from loguru import logger


def load_and_filter_data(
    filepath: Path, model_name: str, channels: list[str], include: bool
) -> pd.DataFrame:
    """Load CSV data and filter by specific channels.

    Args:
        filepath: Path to the CSV file.
        model_name: The name of the model being processed.
        channels: List of channel names to filter by.
        include: If True, keep only the specified channels. If False, exclude them.

    Returns:
        A DataFrame with the filtered channels and renamed metric columns.
    """
    logger.info(f"Loading data from {filepath} for model {model_name}")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        raise

    if "wavelength" not in df.columns:
        error_msg = f"Missing 'wavelength' column in {filepath}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if include:
        df = df[df["wavelength"].isin(channels)].copy()
    else:
        df = df[~df["wavelength"].isin(channels)].copy()

    # Map original metric names to the requested capitalized format
    metric_map = {
        "mse": "MSE",
        "rmse_intensity": "RMSE",
        "mae": "MAE",
        "r2_score": "R-squared",
    }

    # Ensure all required metrics exist before filtering
    available_metrics = [col for col in metric_map if col in df.columns]

    # Keep only wavelength and required metrics
    columns_to_keep = ["wavelength"] + available_metrics
    df = df[columns_to_keep]

    # Rename to the capitalized metric names
    rename_dict = {orig: new for orig, new in metric_map.items() if orig in available_metrics}
    df = df.rename(columns=rename_dict)

    # Prefix the metric columns with the model name to avoid collision during merge
    prefix_rename = {col: f"{model_name}_{col}" for col in df.columns if col != "wavelength"}
    df = df.rename(columns=prefix_rename)

    return df


def create_comparison_table(
    models_dict: dict[str, Path],
    target_channels: list[str],
    include: bool,
    output_path: Path,
) -> None:
    """Create a comparison table across multiple models and compute means.

    Args:
        models_dict: Dictionary mapping model names to their CSV file paths.
        target_channels: List of channels to filter on.
        include: Whether to include (True) or exclude (False) the target_channels.
        output_path: Path to save the final merged CSV.
    """
    merged_df = None

    for model_name, filepath in models_dict.items():
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}. Skipping model {model_name}.")
            continue

        df = load_and_filter_data(filepath, model_name, target_channels, include)

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="wavelength", how="outer")

    if merged_df is None or merged_df.empty:
        logger.warning("No data was merged. Skipping output generation.")
        return

    # Calculate the mean for each numeric column
    mean_series = merged_df.mean(numeric_only=True)
    mean_row = pd.DataFrame([mean_series], columns=mean_series.index)
    mean_row["wavelength"] = "mean"

    # Append the mean row using pd.concat instead of append
    merged_df = pd.concat([merged_df, mean_row], ignore_index=True)

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved comparison table to {output_path}")


def main() -> None:
    """Main execution entrypoint."""
    parser = argparse.ArgumentParser(
        description="Analyze and aggregate model metrics for AIA and HMI channels."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("assets/ds_results/missing_channel"),
        help="Directory containing the input CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/ds_results/missing_channel/summary"),
        help="Directory to save the resulting summary CSV files.",
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    logger.info(f"Starting analysis with input_dir={input_dir} and output_dir={output_dir}")

    # AIA Configuration
    aia_models = {
        "SDOFMv2-aia": input_dir / "missing_channel_test_sdofmv2-aia.csv",
        "SDOFMv2-all-denoise": input_dir / "missing_channel_test_sdofmv2-all-denoise.csv",
        "SDOFMv2-all": input_dir / "missing_channel_test_sdofmv2-all.csv",
        "SDOFMv1": input_dir / "missing_channel_test_results_sdofmv1.csv",
    }
    hmi_channels = ["Bx", "By", "Bz"]
    aia_output_path = output_dir / "aia_comparison_summary.csv"

    # HMI Configuration
    hmi_models = {
        "SDOFMv2-all-denoise": input_dir / "missing_channel_test_sdofmv2-all-denoise.csv",
        "SDOFMv2-all": input_dir / "missing_channel_test_sdofmv2-all.csv",
        "SDOFMv2-HMI": input_dir / "missing_channel_test_sdofmv2-hmi-log.csv",
    }
    hmi_output_path = output_dir / "hmi_comparison_summary.csv"

    # Process AIA
    logger.info("Processing AIA comparisons...")
    create_comparison_table(
        models_dict=aia_models,
        target_channels=hmi_channels,
        include=False,
        output_path=aia_output_path,
    )

    # Process HMI
    logger.info("Processing HMI comparisons...")
    create_comparison_table(
        models_dict=hmi_models,
        target_channels=hmi_channels,
        include=True,
        output_path=hmi_output_path,
    )

    logger.info("Analysis completed successfully.")


if __name__ == "__main__":
    main()
