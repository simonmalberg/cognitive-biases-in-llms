from tests import TestCase, Template, DecisionResult
from script import get_all_biases
from utils import get_model, get_metric
import pandas as pd


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


def decide_dataset(
    model_name: str,
    biases: list[str] = get_all_biases(),
    path_to_dataset: str = "dataset.csv",
    shuffle_answer_options: bool = False,
    temperature: float = 0.7,
    seed: int = 42,
) -> int:
    """
    Decides the dataset using the specified model and saves the results to the output path.

    Args:
    model_name (str): the model to use
    biases (list[str]): the biases to decide
    path_to_dataset (str): the path to the input dataset
    shuffle_answer_options (bool): whether to shuffle the answer options
    temperature (float): the temperature to use
    seed (int): the seed to use

    Returns:
    int: 0 if successful
    """
    # loading the dataset
    dataset = pd.read_csv(path_to_dataset)
    # initializing the model
    model = get_model(model_name, shuffle_answer_options=shuffle_answer_options)
    # initialize decision dataset
    decision_dataset = None
    # iterating over all required biases
    for bias in biases:
        print(f"Deciding for bias: {bias}")
        test_cases, ids = [], []
        # constructing test cases for all relevant rows in the dataset
        for i, row in dataset[dataset["bias"] == bias].iterrows():
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
        decision_results = model.decide_all(test_cases, temperature, seed)
        # storing the results in a new dataset
        decision_df = convert_decisions(ids, decision_results)
        # calculating the metrics
        # metric = get_metric(bias)(test_results=list(zip(test_cases, decision_results)))
        # compute individual and aggregated scores
        # individual_scores = metric.compute()
        # aggregated_score = metric.aggregate(individual_scores)
        # TODO: change this to the actual metric calculation above when the metrics are merged:
        metric = get_metric(bias.replace(" ", ""))
        individual_scores = metric.compute(list(zip(test_cases, decision_results)))
        aggregated_score = individual_scores
        # store the results
        decision_df.loc[:, "individual_score"] = individual_scores
        decision_df.loc[:, "aggregated_score"] = aggregated_score
        decision_df.loc[:, "bias"] = bias

        # appending this bias' decisions with the overall decision dataset
        decision_dataset = (
            decision_df
            if decision_dataset is None
            else pd.concat([decision_dataset, decision_df], ignore_index=True)
        )

    # saving the dataset
    decision_dataset.to_csv(f"dataset_decided_{model_name}.csv", index=False)

    return 0


if __name__ == "__main__":

    _ = decide_dataset(
        model_name="GPT-3.5-Turbo",
        biases=["FramingEffect", "EndowmentEffect"],
        path_to_dataset="dataset.csv",
        shuffle_answer_options=True,
        temperature=0.7,
        seed=42,
    )
