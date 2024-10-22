import sys
import os

# Add the project root directory to sys.path to be able to import functionality from core/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import argparse
from test_check import get_generated_biases


# References to relevant data directories
DATASETS_PATH = os.path.join(".", "data", "generated_datasets")
FINAL_DATASET_PATH = os.path.join(".", "data")
FINAL_DATASET_FILENAME = "full_dataset.csv"


def merge_datasets(datasets_directory: str, output_directory: str, file_name: str) -> pd.DataFrame:
    """
    Merges a set of separate datasets into a single dataset and stores the latter as a CSV file.

    Args:
        datasets_directory (str): The name of the directory where the separate datasets are stored.
        output_directory (str): The name of the directory where the final dataset should be stored.
        file_name (str): The name of the file to save the dataset as (must end with .csv).
    """

    # Create the directory, if it does not yet exist
    os.makedirs(output_directory, exist_ok=True)

    # Join the directory and file names into the full file path
    file_path = os.path.join(output_directory, file_name)

    # Load and concatenate all CSV files from the datasets directory
    dataframes = []
    for csv_file in os.listdir(datasets_directory):
        if csv_file.endswith("_dataset.csv"):
            csv_path = os.path.join(datasets_directory, csv_file)
            df = pd.read_csv(csv_path)
            dataframes.append(df)

    # Concatenate all the loaded DataFrames into a single DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Add a new index column named 'id'
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.index.name = 'id'
    merged_df.reset_index(inplace=True)

    # Store the final DataFrame as a CSV file
    merged_df.to_csv(file_path, index=False)

    print(f"Dataset is successfully assembled and saved in {file_path}")

    return merged_df


def main():
    """
    The main function of this script that loads separate datasets and combines them into a single large dataset.
    """

    # Define a command line argument parser
    parser = argparse.ArgumentParser(description="This script combines separate datasets with generated test case instances for multiple biases into a single large dataset.")
    parser.add_argument("--source_dir", type=str, help="The source directory containing the separate datasets.", default=DATASETS_PATH)
    parser.add_argument("--target_dir", type=str, help="The target directory in which to store the final combined dataset.", default=FINAL_DATASET_PATH)
    parser.add_argument("--file_name", type=str, help="The file name of the final combined dataset (must end with .csv).", default=FINAL_DATASET_FILENAME)
    args = parser.parse_args()
        
    merge_datasets(args.source_dir, args.target_dir, args.file_name)


if __name__ == "__main__":
    main()