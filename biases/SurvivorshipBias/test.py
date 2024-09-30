from base import TestGenerator, LLM, RatioScaleMetric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np
import random


class SurvivorshipBiasTestGenerator(TestGenerator):
    """
    Test generator for the Survivorship Bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Survivorship Bias"
        self.config: TestConfig = super().load_config(self.BIAS)

    def sample_custom_values(self, num_instances: int, iteration_seed: int) -> dict:
        """
        Sample custom values for the test case generation.

        Args:
            num_instances (int): The number of instances expected to be generated for each scenario.
            iteration_seed (int): The seed to use for sampling the custom values.

        Returns:
            dict: A dictionary containing the sampled custom values.
        """

        # Get the value range from the custom values configuration
        custom_values = self.config.get_custom_values()
        range_min, range_max = self.config.get_custom_values()["percentage_range"]
        range_min, range_max = int(range_min), int(range_max)

        # Randomly sample pairs of percentage values signaling how common a certain characteristic is in the survivor and non-survivor group, respectively
        random.seed(iteration_seed)
        survivor_percentage = [random.randint(range_min, range_max) for _ in range(num_instances)]
        non_survivor_percentage = [random.randint(range_min, range_max) for _ in range(num_instances)]

        # Return a dictionary with the sampled values
        return {
            "survivor_percentage": survivor_percentage,
            "non_survivor_percentage": non_survivor_percentage
        }

    def generate(self, model: LLM, scenario: str, custom_values: dict = {}, temperature: float = 0.0, seed: int = 42) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # Insert the percentages into the templates
        control.insert("survivor_percentage", str(custom_values["survivor_percentage"]), "user")
        control.insert("non_survivor_percentage", str(custom_values["non_survivor_percentage"]), "user")
        treatment.insert("survivor_percentage", str(custom_values["survivor_percentage"]), "user")

        # Populate the templates using the model and the scenario
        control, treatment = super().populate(model, control, treatment, scenario)

        # Create a test case object
        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=treatment,
            generator=model.NAME,
            temperature=temperature,
            seed=seed,
            scenario=scenario,
            variant=None,
            remarks=None
        )

        return test_case


class SurvivorshipBiasMetric(RatioScaleMetric):
    """
    A metric that measures the presence and strength of Survivorship Bias based on a set of test results.

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results, k=np.array([1]))