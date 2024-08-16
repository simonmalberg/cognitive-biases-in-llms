import wandb
import os
import random
import xml.etree.ElementTree as ET
from utils import get_generator, get_metric
from tests import TestCase, DecisionResult
from models.OpenAI.gpt import GptThreePointFiveTurbo, GptFourO


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
        seed (int): The seed used for this experiment.
        temperature (float): The temperature used for this experiment.
        shuffle_answer_options (bool): Whether the answer options should be randomly shuffled when making a decision.
    """

    def __init__(self, bias: str, seed: int = 42, temperature: float = 0.7, shuffle_answer_options: bool = False):
        if not "WANDB_KEY" in os.environ:
            raise ValueError("Cannot track experiments due to missing Weights & Biases API key. Please store your API key in environment variable 'WANDB_KEY'.")

        self.bias = bias
        self.seed = seed
        self.temperature = temperature      # TODO Temperature is ignored so far
        self.shuffle_answer_options = shuffle_answer_options

    def get_random_scenario(self):
        # Load the pre-defined scenario strings
        with open("scenarios.txt") as f:
            scenarios = f.readlines()

        # Format the scenario strings by removing any markdown
        scenarios = [s.strip().replace("**", "") for s in scenarios]

        # Randomly pick a scenario
        scenario = random.choice(scenarios)

        return scenario

    def generate(self, scenario: str) -> TestCase:
        # Load the test generator for the bias
        generator = get_generator(self.bias)

        # Instantiate the population LLM
        population_model = GptFourO()       # TODO Allow selection of different LLMs, e.g., via passing the LLM's name as string and having a get_model() method

        # TODO Maybe we can even log the raw LLM inputs and outputs. Can we have an internal prompt log inside the LLM class?

        # Generate a test case
        test_case = generator.generate_all(population_model, [scenario], self.seed)[0]    # TODO Make sure this only returns one test case, e.g., by passing a list with a single variant to generate_all

        # Track the run on Weights & Biases
        run_name = f"Generation {test_case.BIAS}"
        self.init(run_name)
        logs = self.create_test_case_logs(test_case)
        self.log(logs)
        self.finish()

        return test_case

    def decide(self, test_case: TestCase) -> DecisionResult:
        # Load the metric for the bias
        metric = get_metric(self.bias)
        
        # Instantiate the decision LLM
        decision_model = GptThreePointFiveTurbo(shuffle_answer_options=self.shuffle_answer_options)

        # Obtain a decision for the test case
        decision_result = decision_model.decide(test_case=test_case, temperature=self.temperature, seed=self.seed)

        # Calculate the biasedness
        biasedness = metric.compute([(test_case, decision_result)])

        # Track the run on Weights & Biases
        run_name = f"Decision {test_case.BIAS}"
        self.init(run_name)
        logs = self.create_decision_logs(test_case=test_case, decision_result=decision_result, biasedness=biasedness)
        self.log(logs)
        self.finish()

    def create_test_case_logs(self, test_case: TestCase) -> dict:
        # TODO Maybe the TestCase class should have a to_dict() function that returns a dictionary representation of the test case
        # TODO Temperature and seed should be stored inside the TestCase object for later reference
        logs = {
            "Type": "GENERATION",
            "Bias": test_case.BIAS,
            "Generator": test_case.GENERATOR,
            "Scenario": test_case.SCENARIO,
            "Variant": test_case.VARIANT,
            "Control": test_case.CONTROL.format(),
            "Treatment": test_case.TREATMENT.format(),
            "Control (Raw)": ET.tostring(test_case.CONTROL._data),    # TODO Implement parsing/serialization functionality in Template class
            "Treatment (Raw)": ET.tostring(test_case.TREATMENT._data),
            "Remarks": test_case.REMARKS
        }

        return logs

    def create_decision_logs(self, test_case: TestCase, decision_result: DecisionResult, biasedness: float) -> dict:
        # TODO Maybe the DecisionResult class should have a to_dict() function that returns a dictionary representation of the decision result

        # Create the logs for the test case that was used
        logs = self.create_test_case_logs(test_case)

        # Add the fields from the decision result to the logs
        logs["Type"] = "DECISION"
        logs["Decision Model"] = decision_result.MODEL
        logs["Decision Temperature"] = decision_result.TEMPERATURE
        logs["Decision Seed"] = decision_result.SEED
        logs["Shuffled Answer Options"] = self.shuffle_answer_options

        logs["Control Options"] = decision_result.CONTROL_OPTIONS
        logs["Control Option Order"] = decision_result.CONTROL_OPTION_ORDER
        logs["Control Answer"] = decision_result.CONTROL_ANSWER
        logs["Control Decision"] = decision_result.CONTROL_DECISION

        logs["Treatment Options"] = decision_result.TREATMENT_OPTIONS
        logs["Treatment Option Order"] = decision_result.TREATMENT_OPTION_ORDER
        logs["Treatment Answer"] = decision_result.TREATMENT_ANSWER
        logs["Treatment Decision"] = decision_result.TREATMENT_DECISION

        logs["Biasedness"] = biasedness

        return logs

    def init(self, name: str):
        # Start a new wandb run
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=name,
            config={}
        )

    def log(self, metrics: dict):
        wandb.log(metrics)

    def finish(self):
        wandb.finish()


if __name__ == "__main__":

    # TODO Adapt to logic of current App.py
    experiment = Experiment(bias="StatusQuoBias")
    scenario = experiment.get_random_scenario()
    test_case = experiment.generate(scenario)
    decision_result = experiment.decide(test_case)