import wandb
import os
import random
import xml.etree.ElementTree as ET
from utils import get_generator, get_metric
from tests import TestCase, DecisionResult
from models.OpenAI.gpt import GptThreePointFiveTurbo, GptFourO
from models.Llama.model import LlamaThreePointOneSeventyB


# Login to Weights & Biases for experiment tracking
if "WANDB_KEY" in os.environ:
    wandb.login(key=os.environ["WANDB_KEY"])

WANDB_ENTITY = "simonmalberg"
WANDB_PROJECT = "Cognitive Biases in LLMs"


# TODO somehow create a hash of relevant code status


class Experiment:
    """
    A class providing functionality for running cognitive bias experiments and tracking them on Weights & Biases.

    Attributes:
        bias (str): The name of the cognitive bias tested in this experiment.
    """

    def __init__(self, bias: str):
        if not "WANDB_KEY" in os.environ:
            raise ValueError("Cannot track experiments due to missing Weights & Biases API key. Please store your API key in environment variable 'WANDB_KEY'.")

        self.bias = bias

    def get_random_scenario(self) -> str:
        """
        Samples a random scenario.

        Returns:
            str: The sample scenario string.
        """

        # Load the pre-defined scenario strings
        with open("scenarios.txt") as f:
            scenarios = f.readlines()

        # Format the scenario strings by removing any markdown
        scenarios = [s.strip().replace("**", "") for s in scenarios]

        # Randomly pick a scenario
        scenario = random.choice(scenarios)

        return scenario

    def generate(self, scenario: str, seed: int = 42) -> TestCase:
        """
        Generates a new test case for the specified scenario.

        Args:
            scenario (str): The scenario for which to generate the test case.
            seed (int): The seed for generating the scenario.

        Returns:
            TestCase: The generated test case.
        """

        # Load the test generator for the bias
        generator = get_generator(self.bias)

        # Start tracking of this run
        run_name = f"Generation {generator.BIAS}"
        self._start(run_name)

        # Instantiate the population LLM
        population_model = GptFourO()       # TODO Allow selection of different LLMs, e.g., via passing the LLM's name as string and having a get_model() method

        # TODO Maybe we can even log the raw LLM inputs and outputs. Can we have an internal prompt log inside the LLM class?

        # Generate a test case
        test_case = generator.generate_all(population_model, [scenario], seed)[0]    # TODO Make sure this only returns one test case, e.g., by passing a list with a single variant to generate_all

        # Track the run results and finish the run
        logs = self._create_test_case_logs(test_case)
        self._log(logs)
        self._finish()

        return test_case

    def decide(self, test_case: TestCase, seed: int = 42, shuffle_answer_options: bool = False) -> tuple[DecisionResult, float]:
        """
        Obtains a decision for a test case and calculates the biasedness.

        Args:
            test_case (TestCase): The test case defining the decision task.
            seed (int): The seed to be used for obtaining the decision.
            shuffle_answer_options (bool): If True, answer options in the test case will be randomly shuffled.

        Returns:
            tuple[DecisionResult, float]: A tuple containing the decision result and the calculated biasedness.
        """

        # Start tracking of this run
        run_name = f"Decision {test_case.BIAS}"
        self._start(run_name)

        # Load the metric for the bias
        metric = get_metric(self.bias)        

        # Instantiate the decision LLM
        decision_model = LlamaThreePointOneSeventyB(shuffle_answer_options=shuffle_answer_options)

        # Obtain a decision for the test case
        decision_result = decision_model.decide(test_case=test_case, seed=seed)

        # Calculate the biasedness
        biasedness = metric.compute([(test_case, decision_result)])

        # Track the run results and finish the run
        logs = self._create_decision_logs(test_case=test_case, decision_result=decision_result, biasedness=biasedness, shuffled_answer_options=shuffle_answer_options)
        self._log(logs)
        self._finish()

        return decision_result, biasedness

    def _create_test_case_logs(self, test_case: TestCase, shuffled_answer_options: bool = False, seed: int = 42) -> dict:
        """
        Function to be used after the generation of a test case. Converts a TestCase object into run logs.

        Args:
            test_case (TestCase): The TestCase that was generated during the run.
            shuffled_answer_options (bool): Whether the answer options were randomly shuffled for this run.
            seed (int): The seed used for shuffling the answer options.

        Returns:
            dict: A dictionary of run logs.
        """

        # TODO Temperature and seed should be stored inside the TestCase object for later reference
        logs = {
            "Type": "GENERATION",
            "Bias": test_case.BIAS,
            "Generator": test_case.GENERATOR,
            "Scenario": test_case.SCENARIO,
            "Variant": test_case.VARIANT,
            "Control": test_case.CONTROL.format(shuffle_options=shuffled_answer_options, seed=seed),
            "Treatment": test_case.TREATMENT.format(shuffle_options=shuffled_answer_options, seed=seed),
            "Control (Raw)": ET.tostring(test_case.CONTROL._data),    # TODO Implement parsing/serialization functionality in Template class
            "Treatment (Raw)": ET.tostring(test_case.TREATMENT._data),
            "Remarks": test_case.REMARKS
        }

        return logs

    def _create_decision_logs(self, test_case: TestCase, decision_result: DecisionResult, biasedness: float, shuffled_answer_options: bool) -> dict:
        """
        Function to be used after the decision result has been obtained for a test case. Converts a TestCase and a DecisionResult objects together into run logs.

        Args:
            test_case (TestCase): The TestCase that was used during the run.
            decision_result (DecisionResult): The DecisionResult object containing the final model decisions from the run.
            biasedness (float): The biasedness value calculated by the bias metric.
            shuffled_answer_options (bool): Whether the answer options were randomly shuffled for this run.

        Returns:
            dict: A dictionary of run logs.
        """

        # Create the logs for the test case that was used
        logs = self._create_test_case_logs(test_case, shuffled_answer_options, seed=decision_result.SEED)

        # Add the fields from the decision result to the logs
        logs["Type"] = "DECISION"
        logs["Decision Model"] = decision_result.MODEL
        logs["Decision Temperature"] = decision_result.TEMPERATURE
        logs["Decision Seed"] = decision_result.SEED
        logs["Shuffled Answer Options"] = shuffled_answer_options

        logs["Control Options"] = decision_result.CONTROL_OPTIONS
        logs["Control Option Shuffling"] = decision_result.CONTROL_OPTION_SHUFFLING
        logs["Control Answer"] = decision_result.CONTROL_ANSWER
        logs["Control Decision"] = decision_result.CONTROL_DECISION

        logs["Treatment Options"] = decision_result.TREATMENT_OPTIONS
        logs["Treatment Option Shuffling"] = decision_result.TREATMENT_OPTION_SHUFFLING
        logs["Treatment Answer"] = decision_result.TREATMENT_ANSWER
        logs["Treatment Decision"] = decision_result.TREATMENT_DECISION

        logs["Biasedness"] = biasedness

        return logs

    def _start(self, name: str):
        """
        Starts tracking of a new run on Weights & Biases

        Args:
            name (str): Name of the new run.
        """

        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=name
        )

    def _log(self, metrics: dict):
        """
        Logs some metrics to Weights & Biases for the current run.

        Args:
            metrics (dict): A dictionary of metrics to log to the current run.
        """

        wandb.log(metrics)

    def _finish(self):
        """
        Finishes the tracked run on Weights & Biases.
        """

        wandb.finish()


if __name__ == "__main__":

    # TODO Adapt to logic of current App.py
    experiment = Experiment(bias="StatusQuoBias")
    scenario = experiment.get_random_scenario()
    test_case = experiment.generate(scenario)
    decision_result = experiment.decide(test_case)