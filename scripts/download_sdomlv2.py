import s3fs
from pathlib import Path
import argparse


def download_s3_folder(s3_folder: str, local_folder: str, component: str = "both"):
    """
    Download SDOMLv2 Zarr data from a public S3 bucket, skipping files that
    already exist locally with the correct size (resumable).
    """
    fs = s3fs.S3FileSystem(anon=True)
    local_path = Path(local_folder)
    local_path.mkdir(parents=True, exist_ok=True)

    component_files = {
        "aia": ["sdomlv2.zarr"],
        "hmi": ["sdomlv2_hmi.zarr"],
        "both": ["sdomlv2.zarr", "sdomlv2_hmi.zarr"],
    }

    target_zarrs = component_files.get(component.lower(), [])
    for zarr_name in target_zarrs:
        s3_src = f"{s3_folder.rstrip('/')}/{zarr_name}"
        print(f"Scanning {zarr_name}...")

        all_files = fs.find(s3_src)
        skipped, downloaded, failed = 0, 0, 0

        for s3_file in all_files:
            relative = s3_file[len(s3_src) :].lstrip("/")
            local_file = local_path / zarr_name / relative
            local_file.parent.mkdir(parents=True, exist_ok=True)

            # Skip if file exists and size matches
            if local_file.exists():
                remote_size = fs.info(s3_file)["size"]
                if local_file.stat().st_size == remote_size:
                    skipped += 1
                    continue
                print(f"Re-downloading (size mismatch): {relative}")

            try:
                fs.get(s3_file, str(local_file))
                downloaded += 1
            except Exception as e:
                print(f"  ERROR downloading {relative}: {e}")
                failed += 1

        print(
            f"{zarr_name} done — {downloaded} downloaded, {skipped} skipped, {failed} failed."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download SDOMLv2 data from NASA HDRL S3"
    )
    parser.add_argument(
        "--target", type=str, required=True, help="Local directory to save data"
    )
    parser.add_argument(
        "--component",
        type=str,
        default="both",
        choices=["aia", "hmi", "both"],
        help="Which dataset component to download",
    )
    args = parser.parse_args()

    S3_BUCKET_PATH = "gov-nasa-hdrl-data1/contrib/fdl-sdoml/fdl-sdoml-v2"
    download_s3_folder(S3_BUCKET_PATH, args.target, args.component)
