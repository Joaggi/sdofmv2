"""Script to analyze and compare test results for different models."""

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

# Constants for file mapping
AIA_MODELS = {
    "sdofmv1": "sdofmv1-test-results.csv",
    "sdofmv2-aia": "sdofmv2-aia-test-results-denoise.csv",
    "sdofmv2": "sdofmv2-all-test-results-denoise.csv",
    "sdofmv2_large": "sdofmv2-all-test-results-denoise-large-final.csv",
    "sdofmv2_small": "sdofmv2-all-test-results-denoise-small-final.csv",
}

HMI_MODELS = {
    "sdofmv2": "sdofmv2-all-test-results-denoise.csv",
    "sdofmv2_hmi": "sdofmv2-hmi-test-results-denoise-final.csv",
    "sdofmv2_large": "sdofmv2-all-test-results-denoise-large-final.csv",
    "sdofmv2_small": "sdofmv2-all-test-results-denoise-small-final.csv",
}

AIA_CHANNELS = [
    "94A",
    "131A",
    "171A",
    "193A",
    "211A",
    "304A",
    "335A",
    "1600A",
    "1700A",
]
HMI_CHANNELS = ["Bx", "By", "Bz"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Analyze model test results.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./assets/test_results",
        help="Directory containing the test results CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./assets/analysis_output",
        help="Directory to save the analysis results.",
    )
    return parser.parse_args()


def load_and_filter_data(
    results_dir: Path, model_mapping: dict[str, str], channels: list[str]
) -> pd.DataFrame:
    """Load data for specified models and filter by channels.

    Args:
        results_dir (Path): The directory containing the CSV files.
        model_mapping (dict[str, str]): Mapping from model name to filename.
        channels (list[str]): List of channels to filter by.

    Returns:
        pd.DataFrame: A concatenated dataframe with a 'model' column.
    """
    dataframes = []
    for model_name, filename in model_mapping.items():
        file_path = results_dir / filename
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        df_filtered = df[df["channel"].isin(channels)].copy()
        df_filtered["model"] = model_name
        dataframes.append(df_filtered)

    if not dataframes:
        raise ValueError("No data could be loaded for the given models.")

    return pd.concat(dataframes, ignore_index=True)


def create_comparison_pivot(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Create a pivot table for a specific metric and save to CSV.

    Args:
        df (pd.DataFrame): The combined dataframe.
        metric (str): The metric column to pivot.
        output_path (Path): Path to save the pivot table CSV.
    """
    if metric not in df.columns:
        logger.warning(f"Metric {metric} not found in dataframe.")
        return

    pivot_df = df.pivot(index="channel", columns="model", values=metric)
    pivot_df.loc["mean"] = pivot_df.mean()
    pivot_df.to_csv(output_path)
    logger.info(f"Saved {metric} comparison to {output_path}")


def analyze_group(
    results_dir: Path,
    output_dir: Path,
    group_name: str,
    model_mapping: dict[str, str],
    channels: list[str],
) -> None:
    """Analyze a specific group of models and channels.

    Args:
        results_dir (Path): Directory with result CSVs.
        output_dir (Path): Directory to save outputs.
        group_name (str): Name of the group (e.g., 'aia', 'hmi').
        model_mapping (dict[str, str]): Model to filename mapping.
        channels (list[str]): Channels to include.
    """
    logger.info(f"Analyzing {group_name} group...")
    df = load_and_filter_data(results_dir, model_mapping, channels)

    # Metrics to compare
    metrics = [
        "mse_norm",
        "rmse_intensity_norm",
        "mae_norm",
        "r2_score_norm",
        "pixel_correlation_norm",
    ]

    for metric in metrics:
        output_file = output_dir / f"{group_name}_comparison_{metric}.csv"
        create_comparison_pivot(df, metric, output_file)


def main() -> None:
    """Main execution function."""
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        analyze_group(results_dir, output_dir, "aia", AIA_MODELS, AIA_CHANNELS)
        analyze_group(results_dir, output_dir, "hmi", HMI_MODELS, HMI_CHANNELS)
        logger.success("Analysis complete.")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
