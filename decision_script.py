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
        "control_extraction": list(zip(*decision_data))[7],
        "control_decision": list(zip(*decision_data))[8],
        "treatment_options": list(zip(*decision_data))[9],
        "treatment_option_order": list(zip(*decision_data))[10],
        "treatment_answer": list(zip(*decision_data))[11],
        "treatment_extraction": list(zip(*decision_data))[12],
        "treatment_decision": list(zip(*decision_data))[13],
        "status": list(zip(*decision_data))[14],
        "error_message": list(zip(*decision_data))[15]
    }
    # storing the results in a new dataset
    decision_df = pd.DataFrame(decision_data)
    return decision_df


def decide_batch(
    batch: pd.DataFrame,
    model_name: str,
    randomly_flip_options: bool,
    shuffle_answer_options: bool,
    temperature: float,
    seed: int,
) -> None:
    """
    Decides the dataset batch using the specified model.

    Args:
    batch (pd.DataFrame): the batch of the dataset to decide
    model_name (str): the model to use
    randomly_flip_options (bool): whether to reverse the answer options
    shuffle_answer_options (bool): whether to shuffle the answer options
    temperature (float): the temperature to use
    seed (int): the seed to use

    Returns:
    None
    """
    # initializing the model
    model = get_model(model_name, randomly_flip_options=randomly_flip_options, shuffle_answer_options=shuffle_answer_options)
    # initialize decision batch
    decision_batch = None
    # identify the biases in the batch
    biases = [''.join(' ' + char if char.isupper() else char for char in bias).strip().title().replace(' ', '') for bias in batch["bias"].unique()]
    # iterating over all required biases
    for bias in biases:
        test_cases, ids = [], []
        # constructing test cases for all relevant rows in the batch
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
        # deciding the test cases and obtaining the DecisionResult objects
        decision_results = model.decide_all(test_cases, temperature, seed, max_retries=5)
        # storing all the results (both failed and completed) in a new DataFrame
        decision_df = convert_decisions(ids, decision_results)
        # get indices of the decisions that failed: they have status "ERROR"
        failed_idx = [i for i, decision_result in enumerate(decision_results) if decision_result.STATUS == "ERROR"]
        # removing failed decisions from the decision results to calculate the metric
        decision_results = [decision_result for i, decision_result in enumerate(decision_results) if i not in failed_idx]
        # remove corresponding test cases to calculate the metric
        test_cases = [test_case for i, test_case in enumerate(test_cases) if i not in failed_idx]
        # calculating the metrics if we have any correct decisions
        if len(test_cases) > 0 and len(decision_results) > 0:
            metric = get_metric(bias)(
                test_results=list(zip(test_cases, decision_results))
            )
            # compute individual and aggregated scores
            individual_scores = metric.compute()
            # store the results in the rows of the "OK" decisions
            decision_df.loc[decision_df['status'] == "OK", "individual_score"] = individual_scores
            # store the weights of the individual scores of "OK" decisions
            decision_df.loc[decision_df['status'] == "OK", "weight"] = metric.test_weights
        decision_df.loc[:, "bias"] = bias

        # appending this bias' decisions with the overall decision for the batch
        decision_batch = (
            decision_df
            if decision_batch is None
            else pd.concat([decision_batch, decision_df], ignore_index=True)
        )
    
    # saving the decisions for the batch to a CSV file with a unique name based on the process ID and timestamp
    decision_batch.to_csv(
        f"decision_datasets/{model_name}/batch_{os.getpid()}_decided_{datetime.datetime.now()}.csv",
        index=False,
    )


def decide_dataset(
    batches: list[pd.DataFrame],
    model_name: str,
    num_workers: int,
    randomly_flip_options: bool,
    shuffle_answer_options: bool,
    temperature: float,
    seed: int,
) -> None:
    """
    Function that encapsulates the parallel decision making process for a dataset.

    Args:
    batches (list[pd.DataFrame]): the batches of the dataset to decide (of length num_workers)
    model_name (str): the name of the model to use
    num_workers (int): the number of workers used
    randomly_flip_options (bool): whether to reverse the answer options
    shuffle_answer_options (bool): whether to shuffle the answer options
    temperature (float): the temperature to use
    seed (int): the seed to use
    """
    # Prepare the directory to store the overall decision dataset for the model
    os.makedirs(f"decision_datasets/{model_name}", exist_ok=True)
    
    with tqdm(total=len(batches)) as progress_bar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executer:
            # Decide the dataset in parallel
            for _ in executer.map(
                partial(
                    decide_batch,
                    model_name=model_name,
                    randomly_flip_options=randomly_flip_options,
                    shuffle_answer_options=shuffle_answer_options,
                    temperature=temperature,
                    seed=seed,
                ),
                batches,
            ):
                progress_bar.update()


if __name__ == "__main__":

    # TODO: name of the decision model as per the get_model function
    model_name = 'Llama-3.1-8B'
    # TODO: Decide the number of batches to split the dataset into:
    N_BATCHES = 1#3000 # 10 tests per batch
    # TODO: Number of processors to use
    max_workers = 1

    # Provide the path to the overall dataset if location is different from the default
    dataset = pd.read_csv("datasets/full_dataset.csv").sample(n=20)
    print(f"Number of workers used: {max_workers}")
    # Preparing the batches
    batches = np.array_split(dataset, N_BATCHES)
    # Deciding the dataset
    print("Starting the decision making process...")
    start_time = datetime.datetime.now()
    _ = decide_dataset(
        batches=batches,
        model_name=model_name,
        num_workers=max_workers,
        randomly_flip_options=True,
        shuffle_answer_options=False,
        temperature=0.0,
        seed=42,
    )
    print(
        f"Decisions for the model {model_name} completed in {datetime.datetime.now() - start_time} seconds."
    )
