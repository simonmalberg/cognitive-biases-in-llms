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


# TODO somehow create a hash of relevant code status


class Experiment:

    def __init__(self, bias: str, seed: int = 42, temperature: float = 0.7, shuffle_answer_options: bool = False):
        if not "WANDB_KEY" in os.environ:
            raise ValueError("Cannot track experiments due to missing Weights & Biases API key. Please store your API key in environment variable 'WANDB_KEY'.")

        self.WANDB_ENTITY = "simonmalberg"
        self.WANDB_PROJECT = "Cognitive Biases in LLMs"

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
        test_cases = generator.generate_all(population_model, [scenario], self.seed)    # TODO Make sure this only returns one test case, e.g., by passing a list with a single variant to generate_all
        test_case = test_cases[0]

        # Track the run on Weights & Biases
        run_name = f"Generation {self.bias}"
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
        decision_results = decision_model.decide_all(test_cases=[test_case], temperature=self.temperature, seed=self.seed)
        decision_result = decision_results[0]

        # Calculate the biasedness
        biasedness = metric.compute([(test_case, decision_result)])

        # Track the run on Weights & Biases
        run_name = f"Decision {self.bias}"
        self.init(run_name)
        logs = self.create_decision_logs(test_case=test_case, decision_result=decision_result, biasedness=biasedness)
        self.log(logs)
        self.finish()

    def create_test_case_logs(self, test_case: TestCase) -> dict:
        # TODO Maybe the TestCase class should have a to_dict() function that returns a dictionary representation of the test case
        logs = {
            "run_type": "GENERATION",
            "bias": test_case.BIAS,
            "generator": test_case.GENERATOR,
            "scenario": test_case.SCENARIO,
            "variant": test_case.VARIANT,
            "control": test_case.CONTROL.format(),
            "treatment": test_case.TREATMENT.format(),
            "control_raw": ET.tostring(test_case.CONTROL._data),    # TODO Implement parsing/serialization functionality in Template class
            "treatment_raw": ET.tostring(test_case.TREATMENT._data),
            "remarks": test_case.REMARKS
        }

        return logs

    def create_decision_logs(self, test_case: TestCase, decision_result: DecisionResult, biasedness: float) -> dict:
        # TODO Maybe the DecisionResult class should have a to_dict() function that returns a dictionary representation of the decision result
        logs = self.create_test_case_logs(test_case)
        logs["run_type"] = "DECISION"
        logs["decision_model"] = decision_result.MODEL
        logs["decision_temperature"] = decision_result.TEMPERATURE
        logs["decision_seed"] = decision_result.SEED
        logs["shuffle_anwer_options"] = self.shuffle_answer_options

        logs["control_options"] = decision_result.CONTROL_OPTIONS
        logs["control_option_order"] = decision_result.CONTROL_OPTION_ORDER
        logs["control_answer"] = decision_result.CONTROL_ANSWER
        logs["control_decision"] = decision_result.CONTROL_DECISION

        logs["treatment_options"] = decision_result.TREATMENT_OPTIONS
        logs["treatment_option_order"] = decision_result.TREATMENT_OPTION_ORDER
        logs["treatment_answer"] = decision_result.TREATMENT_ANSWER
        logs["treatment_decision"] = decision_result.TREATMENT_DECISION

        logs["biasedness"] = biasedness

        return logs

    def init(self, name: str):
        # Start a new wandb run
        wandb.init(
            entity=self.WANDB_ENTITY,
            project=self.WANDB_PROJECT,
            name=name,
            config={
                "seed": self.seed,
                "temperature": self.temperature # TODO Temperature and seed should be stored inside the TestCase object for later reference
            }
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