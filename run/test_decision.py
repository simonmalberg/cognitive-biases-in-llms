import sys
import os

# Add the project root directory to sys.path to be able to import functionality from core/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.testing import TestCase, Template, DecisionResult
from core.utils import get_model, get_metric
from dataset_assembly import merge_datasets
import pandas as pd
import concurrent.futures
from functools import partial
import datetime
import numpy as np
from tqdm import tqdm
import argparse


# References to relevant data directories
DATASET_FILE_PATH = os.path.join(".", "data", "full_dataset.csv")
DECISION_RESULTS = os.path.join(".", "data", "decision_results")


def convert_decisions(ids: list[int], decision_results: list[DecisionResult]) -> pd.DataFrame:
    """
    Converts a list of DecisionResult objects into a DataFrame.

    Args:
        ids (list[int]): The ids of the decision results.
        decision_results (list[DecisionResult]): The DecisionResult objects.

    Returns:
        pd.DataFrame: the DataFrame representation of the decision results.
    """

    decision_data = [
        [
            decision_result.MODEL,
            decision_result.TEMPERATURE,
            decision_result.SEED,
            decision_result.TIMESTAMP,
            decision_result.CONTROL_OPTIONS,
            decision_result.CONTROL_OPTION_SHUFFLING,
            decision_result.CONTROL_ANSWER,
            decision_result.CONTROL_EXTRACTION,
            decision_result.CONTROL_DECISION,
            decision_result.TREATMENT_OPTIONS,
            decision_result.TREATMENT_OPTION_SHUFFLING,
            decision_result.TREATMENT_ANSWER,
            decision_result.TREATMENT_EXTRACTION,
            decision_result.TREATMENT_DECISION,
            decision_result.STATUS,
            decision_result.ERROR_MESSAGE
        ]
        for decision_result in decision_results
    ]
    
    # Wrap the results in a new DataFrame
    decision_df = pd.DataFrame({
        "id": ids,
        "model": list(zip(*decision_data))[0],
        "temperature": list(zip(*decision_data))[1],
        "seed": list(zip(*decision_data))[2],
        "timestamp": list(zip(*decision_data))[3],
        "control_options": list(zip(*decision_data))[4],
        "control_option_order": list(zip(*decision_data))[5],
        "control_answer": list(zip(*decision_data))[6],
        "control_extraction": list(zip(*decision_data))[7],
        "control_decision": list(zip(*decision_data))[8],
        "treatment_options": list(zip(*decision_data))[9],
        "treatment_option_order": list(zip(*decision_data))[10],
        "treatment_answer": list(zip(*decision_data))[11],
        "treatment_extraction": list(zip(*decision_data))[12],
        "treatment_decision": list(zip(*decision_data))[13],
        "status": list(zip(*decision_data))[14],
        "error_message": list(zip(*decision_data))[15]
    })

    return decision_df


def decide_batch(batch: pd.DataFrame, model_name: str, randomly_flip_options: bool, shuffle_answer_options: bool, temperature: float, seed: int):
    """
    Decides the dataset batch using the specified model.

    Args:
        batch (pd.DataFrame): The batch of the dataset with generated test case instances to decide.
        model_name (str): The name of the model to use for obtaining decisions.
        randomly_flip_options (bool): Whether to reverse the answer options in 50% of test cases.
        shuffle_answer_options (bool): Whether to shuffle the answer options randomly for all test cases.
        temperature (float): The temperature to use for the decision model.
        seed (int): The seed to use for reproducibility.
    """

    # Get an instance of the model
    model = get_model(model_name, randomly_flip_options=randomly_flip_options, shuffle_answer_options=shuffle_answer_options)

    # Initialize a decision batch
    decision_batch = None

    # Identify the biases in the batch
    biases = [''.join(' ' + char if char.isupper() else char for char in bias).strip().title().replace(' ', '') for bias in batch["bias"].unique()]

    # Iterate over all biases in the batch
    for bias in biases:
        test_cases, ids = [], []

        # Construct test cases for all relevant rows in the batch
        for _, row in batch[batch['bias'].str.strip().str.title().str.replace(' ', '') == bias].iterrows():
            ids.append(row['id'])
            test_cases.append(
                TestCase(
                    bias=row["bias"],
                    control=Template(row["raw_control"]),
                    treatment=Template(row["raw_treatment"]),
                    generator=row["generator"],
                    temperature=row["temperature"],
                    seed=row["seed"],
                    scenario=row["scenario"],
                    variant=row["variant"],
                    remarks=row["remarks"],
                )
            )

        # Decide the test cases and obtain the DecisionResult objects
        decision_results = model.decide_all(test_cases, temperature, seed, max_retries=1)

        # Store all the results (both failed and completed) in a new DataFrame
        decision_df = convert_decisions(ids, decision_results)

        # Get indices of the decisions that failed: they have status "ERROR"
        failed_idx = [i for i, decision_result in enumerate(decision_results) if decision_result.STATUS == "ERROR"]

        # Remove failed decisions from the decision results to calculate the metric
        decision_results = [decision_result for i, decision_result in enumerate(decision_results) if i not in failed_idx]

        # Remove corresponding test cases to calculate the metric
        test_cases = [test_case for i, test_case in enumerate(test_cases) if i not in failed_idx]

        # Calculate the metrics if we have any correct decisions
        if len(test_cases) > 0 and len(decision_results) > 0:
            metric = get_metric(bias)(test_results=list(zip(test_cases, decision_results)))
            individual_scores = metric.compute()

            # Store the results and weights in the rows of the "OK" decisions
            decision_df.loc[decision_df['status'] == "OK", "individual_score"] = individual_scores
            decision_df.loc[decision_df['status'] == "OK", "weight"] = metric.test_weights
        decision_df.loc[:, "bias"] = bias

        # Append this bias's decisions to the overall decisions for the batch
        decision_batch = (
            decision_df
            if decision_batch is None
            else pd.concat([decision_batch, decision_df], ignore_index=True)
        )
    
    # Save the decisions for the batch to a CSV file with a unique name based on the process ID and timestamp
    file_name = os.path.join(DECISION_RESULTS, model_name, f"batch_{os.getpid()}_decided_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    decision_batch.to_csv(file_name, index=False)


def decide_dataset(dataset: pd.DataFrame, model_name: str, n_batches: int, n_workers: int, randomly_flip_options: bool, shuffle_answer_options: bool, temperature: float, seed: int):
    """
    Function that encapsulates the parallel decision-making process for a dataset.

    Args:
        dataset (pd.DataFrame): The dataset with generated test case instances to decide.
        model_name (str): The name of the model to use for obtaining decisions.
        n_batches (int): The number of equally-sized batches to split the dataset into for distribution across the parallel workers.
        n_workers (int): The maximum number of parallel workers used.
        randomly_flip_options (bool): Whether to reverse the answer options in 50% of test cases.
        shuffle_answer_options (bool): Whether to shuffle the answer options randomly for all test cases.
        temperature (float): The temperature to use for the decision model.
        seed (int): The seed to use for reproducibility.
    """

    # Prepare the directory to store the decision results of the model
    results_directory = os.path.join(DECISION_RESULTS, model_name)
    os.makedirs(results_directory, exist_ok=True)

    # Split the dataset into equally-sized batches for distribution across the parallel workers
    batches = np.array_split(dataset, n_batches)
    
    # Allocate the batches to the workers to obtain decisions.
    with tqdm(total=len(batches)) as progress_bar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executer:
            for _ in executer.map(
                partial(
                    decide_batch,
                    model_name=model_name,
                    randomly_flip_options=randomly_flip_options,
                    shuffle_answer_options=shuffle_answer_options,
                    temperature=temperature,
                    seed=seed
                ),
                batches
            ):
                progress_bar.update()

    # Merge all batch results into a single CSV containing all decision results of the model
    merge_datasets(results_directory, DECISION_RESULTS, f"{model_name}.csv", add_id=False)


def main():
    """
    The main function of this script that obtains decisions from models for generated test case instances.
    """

    # Define a command line argument parser
    parser = argparse.ArgumentParser(description="This script obtains decisions from models for generated test case instances.")
    parser.add_argument("--dataset", type=str, help="The path to the dataset file with the test case instances.", default=DATASET_FILE_PATH)
    parser.add_argument("--model", type=str, help="The name of the model to obtain decisions from.", default="GPT-4o-Mini")
    parser.add_argument("--n_workers", type=int, help="The maximum number of parallel workers obtaining decisions from the model.", default=100)
    parser.add_argument("--n_batches", type=int, help="The number of equally-sized batches to split the dataset into to distribute them to the workers.", default=3000)
    parser.add_argument("--temperature", type=float, help="Temperature value of the decision model", default=0.0)
    parser.add_argument("--seed", type=int, help="The seed to use for reproducibility.", default=42)
    args = parser.parse_args()

    # Load the dataset with generated test case instances
    dataset = pd.read_csv(args.dataset)

    # Obtain decisions from the model for all test case instances in the dataset
    print("Starting the decision-making process with {args.n_workers} parallel workers ...")
    start_time = datetime.datetime.now()
    decide_dataset(
        dataset=dataset,
        model_name=args.model,
        n_batches=args.n_batches,
        n_workers=args.n_workers,
        randomly_flip_options=True,
        shuffle_answer_options=False,
        temperature=0.0,
        seed=42,
    )
    print(f"All decisions obtained from model '{args.model}' in {datetime.datetime.now() - start_time} seconds.")


if __name__ == "__main__":
    main()