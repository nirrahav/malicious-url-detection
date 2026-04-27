import pandas as pd
import os
import subprocess

def load_data(file_path):
    return pd.read_csv(file_path)




def ensure_data_exists(
    file_path: str = "data/raw/malicious_phish.csv",
    dataset: str = "sid321axn/malicious-urls-dataset",
):
    """
    Ensures that the dataset exists locally.
    If not, attempts to download it using Kaggle API.

    Args:
        file_path (str): Path where the dataset should exist.
        dataset (str): Kaggle dataset identifier.
    """

    if os.path.exists(file_path):
        print(f"Dataset already exists at: {file_path}")
        return

    print("Dataset not found. Attempting to download from Kaggle...")

    try:
        # Create directory if not exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Download dataset
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset,
                "-p",
                "data/raw",
                "--unzip",
            ],
            check=True,
        )

        print("Download completed successfully.")

    except Exception as e:
        print("Failed to download dataset.")
        print("Please download it manually from Kaggle and place it in data/raw/")
        print(f"Kaggle dataset: https://www.kaggle.com/datasets/{dataset}")
        print(f"Error: {e}")