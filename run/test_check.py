import pandas as pd
import os
import hashlib
import argparse


# References to relevant data directories
DATASETS_PATH = os.path.join(".", "data", "generated_datasets")
CHECKED_PATH = os.path.join(".", "data", "checked_datasets")


def get_generated_biases(dataset_directory: str) -> list[str]:
    """
    This function takes a directory path and returns a list biases for which test case instances where generated and stored in the directory.
    The directory can contain multiple files of different types, but only CSV files named in the format {bias}_dataset.csv will be considered.

    Args:
        dataset_directory (str): The path to the directory containing the dataset files.

    Returns:
        list[str]: A list of unique bias values.
    """

    # Use a set to ensure uniqueness
    biases = set()

    # Iterate over all files in the directory
    for file_name in os.listdir(dataset_directory):
        # Check if the file matches the pattern {bias}_dataset.csv
        if file_name.endswith("_dataset.csv"):
            bias_name = file_name.rsplit("_dataset.csv", 1)[0]
            biases.add(bias_name)

    return list(biases)


def conduct_check(dataset_directory: str, output_directory: str, biases: list[str], n_sample: int = 10, seed: int = 0):
    """
    Samples n_sample generated test case instances and leads the user through an interactive manual quality check of these instances.
    
    Args:
        dataset_directory (str): The directory containing the CSV datasets with generated test case instances.
        output_directory (str): The directory where the checked CSV file will be stored.
        biases (list[str]): The biases to sample test case instances from.
        n_sample (int): The number of test case instances to sample per bias.
        seed (int): The seed for the random sampling.
    """

    # Correct the provided bias names to match the dataset format (i.e., convert to title case)
    biases = [''.join(' ' + char if char.isupper() else char for char in bias).strip().title().replace(' ', '') for bias in biases]

    # Create the directory, if it does not yet exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over all biases and conduct the check for each
    for bias in biases:
        print(f'Start of the review for: {bias}. Answer 1 if the test is correct, 0 if it is not.')

        # Load the dataset with the generated test case instances of that bias
        file_path = os.path.join(dataset_directory, f"{bias}_dataset.csv")
        dataset = pd.read_csv(file_path)
        dataset = dataset[dataset['bias'].str.strip().str.title().str.replace(' ', '') == bias]

        # Compose a sampling seed from the passed seed and the bias name, so that not the same rows will be sampled for all biases
        sampling_seed = (seed + int(hashlib.md5(bias.encode()).hexdigest(), 16)) % (2**32)

        # Sample n_sample generated test case instances from the dataset
        n_sample_iter = n_sample
        if len(dataset) < n_sample:
            print(f"Attempting to sample {n_sample} instances but dataset only contains {len(dataset)}. Setting n_sample={len(dataset)} for this iteration.")
            n_sample_iter = len(dataset)
        sample = dataset.sample(n=n_sample_iter, random_state=sampling_seed)

        # Let the user manually check all sampled test case instances
        counter = 1
        for i, row in sample.reset_index().iterrows():
            # Display the test case instance
            print('-----------------------------------')
            print(f'SCENARIO:\n{row["scenario"]}\n')
            print(f'CONTROL:\n{row["control"]}')
            print(f'TREATMENT:\n{row["treatment"]}')

            # Obtain the correctness check from the user
            correctness = None
            while correctness not in ['1', '0']:
                correctness = input(f'{i+1}/{n_sample_iter}: Correct? (1 - yes/0 - no): ')  
            
            # Add the correctness to the initial dataset
            dataset.loc[i, 'correct'] = int(correctness)
            
            counter += 1

        # Store the checked dataset
        dataset.to_csv(os.path.join(output_directory, f"{bias}_dataset_checked.csv"), index=False)


def main():
    """
    The main function of this script that loads generated test case instances and guides the user to the manual check.
    """

    # Define a command line argument parser
    parser = argparse.ArgumentParser(description="This script guides the user through an interactive manual quality check of generated test case instances.")
    parser.add_argument("--bias", type=str, help="The names of the cognitive biases to sample test case instances from. Separate multiple biases with a comma. If not provided, this will default to all cognitive biases defined in this repository.", default=None)
    parser.add_argument("--n_sample", type=int, help="The number of generated test case instances to sample for the manual check.", default=10)
    parser.add_argument("--seed", type=int, help="The seed to use for randomly sampling test case instances.", default=0)
    args = parser.parse_args()

    # Parse the list of selected biases to sample test case instances from. If none are provided, select all biases implemented in this repository
    all_biases = get_generated_biases(DATASETS_PATH)
    biases = []
    if args.bias is not None:
        biases = [b.strip() for b in args.bias.split(',')]
        biases = [b.title().replace(' ', '') if ' ' in b else b for b in biases]
    if len(biases) == 0:
        biases = all_biases

    # Validate that test case instances have already been generated for all provided biases
    for bias in biases:
        if bias not in all_biases:
            raise ValueError(f"No generated test case instances found for '{bias}'. Please generate test case instances first.")

    # Guide the user through the interactive manual quaility check
    conduct_check(DATASETS_PATH, CHECKED_PATH, biases=biases, n_sample=args.n_sample, seed=args.seed)


if __name__ == '__main__':
    main()