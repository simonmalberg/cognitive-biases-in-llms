from tests import TestCase, Template, DecisionResult
from utils import get_model, get_metric
import pandas as pd
import concurrent.futures
from functools import partial
import datetime
import os
import numpy as np
from tqdm import tqdm


def convert_decisions(
    ids: list[int], decision_results: list[DecisionResult]
) -> pd.DataFrame:
    """
    Converts the decision results to a DataFrame.

    Args:
    ids (list[int]): the ids of the decision results
    decision_results (list[DecisionResult]): the decision results

    Returns:
    pd.DataFrame: the DataFrame representation of the decision results
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
            decision_result.CONTROL_DECISION,
            decision_result.TREATMENT_OPTIONS,
            decision_result.TREATMENT_OPTION_SHUFFLING,
            decision_result.TREATMENT_ANSWER,
            decision_result.TREATMENT_DECISION,
        ]
        for decision_result in decision_results
    ]
    # storing the decisions
    decision_data = {
        "id": ids,
        "model": list(zip(*decision_data))[0],
        "temperature": list(zip(*decision_data))[1],
        "seed": list(zip(*decision_data))[2],
        "timestamp": list(zip(*decision_data))[3],
        "control_options": list(zip(*decision_data))[4],
        "control_option_order": list(zip(*decision_data))[5],
        "control_answer": list(zip(*decision_data))[6],
        "control_decision": list(zip(*decision_data))[7],
        "treatment_options": list(zip(*decision_data))[8],
        "treatment_option_order": list(zip(*decision_data))[9],
        "treatment_answer": list(zip(*decision_data))[10],
        "treatment_decision": list(zip(*decision_data))[11],
    }
    # storing the results in a new dataset
    decision_df = pd.DataFrame(decision_data)
    return decision_df


def decide_batch(
    batch: pd.DataFrame,
    model_name: str,
    reverse_answer_options: bool,
    shuffle_answer_options: bool,
    temperature: float,
    seed: int,
) -> pd.DataFrame:
    """
    Decides the dataset batch using the specified model.

    Args:
    batch (pd.DataFrame): the batch of the dataset to decide
    model_name (str): the model to use
    reverse_answer_options (bool): whether to reverse the answer options
    shuffle_answer_options (bool): whether to shuffle the answer options
    temperature (float): the temperature to use
    seed (int): the seed to use

    Returns:
    pd.DataFrame: the DataFrame representation of the decisions for the batch
    """
    # initializing the model
    model = get_model(model_name, reverse_answer_options=reverse_answer_options, shuffle_answer_options=shuffle_answer_options)
    # initialize decision batch
    decision_batch = None
    # iterating over all required biases
    for bias in batch["bias"].unique():
        test_cases, ids = [], []
        # constructing test cases for all relevant rows in the batch
        for i, row in batch[batch["bias"] == bias].iterrows():
            ids.append(i)
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
        # deciding the test cases and obtaining the DecisionResult objects
        decision_results = model.decide_all(test_cases, temperature, seed, max_retries=5)
        # removing potential failed decisions (None values) from the decision results
        decision_results = [decision_result for decision_result in decision_results if decision_result is not None]
        # storing the results in a new DataFrame
        decision_df = convert_decisions(ids, decision_results)
        # calculating the metrics
        metric = get_metric("".join(bias.split()))(
            test_results=list(zip(test_cases, decision_results))
        )
        # compute individual and aggregated scores
        individual_scores = metric.compute()
        # We are not aggregating the scores as using parallel processing
        # aggregated_score = metric.aggregate(individual_scores)
        # store the results
        decision_df.loc[:, "individual_score"] = individual_scores
        # store the weights of the individual scores
        decision_df.loc[:, "weight"] = metric.test_weights
        # decision_df.loc[:, "aggregated_score"] = aggregated_score
        decision_df.loc[:, "bias"] = bias

        # appending this bias' decisions with the overall decision for the batch
        decision_batch = (
            decision_df
            if decision_batch is None
            else pd.concat([decision_batch, decision_df], ignore_index=True)
        )

    return decision_batch


def decide_dataset(
    batches: list[pd.DataFrame],
    model_name: str,
    num_processors: int,
    reverse_answer_options: bool,
    shuffle_answer_options: bool,
    temperature: float,
    seed: int,
) -> None:
    """
    Function that encapsulates the parallel decision making process for a dataset.

    Args:
    batches (list[pd.DataFrame]): the batches of the dataset to decide (of length num_processors)
    model_name (str): the name of the model to use
    num_processors (int): the number of processors used
    reverse_answer_options (bool): whether to reverse the answer options
    shuffle_answer_options (bool): whether to shuffle the answer options
    temperature (float): the temperature to use
    seed (int): the seed to use
    """
    results = []
    with tqdm(total=len(batches)) as progress_bar:
        with concurrent.futures.ProcessPoolExecutor(num_processors) as executer:
            # Decide the dataset in parallel
            for result in executer.map(
                partial(
                    decide_batch,
                    model_name=model_name,
                    reverse_answer_options=reverse_answer_options,
                    shuffle_answer_options=shuffle_answer_options,
                    temperature=temperature,
                    seed=seed,
                ),
                batches,
            ):
                results.append(result)
                progress_bar.update()
    # Concatenate the results
    decision_dataset = pd.concat(results)
    # Prepare the directory to store the overall decision dataset for the model
    os.makedirs(f"decision_datasets", exist_ok=True)
    # Write the dataset to a CSV file
    decision_dataset.to_csv(
        f"decision_datasets/dataset_decided_{model_name}_{datetime.datetime.now()}.csv",
        index=False,
    )


if __name__ == "__main__":

    # TODO: name of the decision model as per the get_model function
    model_name = "GPT-3.5-Turbo"
    # TODO: Decide the number of batches to split the dataset into:
    N_BATCHES = 3000

    # Provide the path to the overall dataset if location is different from the default
    dataset = pd.read_csv("datasets/dataset.csv")
    # Number of processors to use
    processors = os.cpu_count()
    print(f"Number of processors used: {processors}")
    # Preparing the batches
    batches = np.array_split(dataset, N_BATCHES)
    # Deciding the dataset
    print("Starting the decision making process...")
    start_time = datetime.datetime.now()
    _ = decide_dataset(
        batches=batches,
        model_name=model_name,
        num_processors=processors,
        reverse_answer_options=True,
        shuffle_answer_options=False,
        temperature=0.0,
        seed=42,
    )
    print(
        f"Decisions for the model {model_name} completed in {datetime.datetime.now() - start_time} seconds."
    )
