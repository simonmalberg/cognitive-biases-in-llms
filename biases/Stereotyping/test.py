from base import TestGenerator, LLM, RatioScaleMetric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np
import random


class StereotypingTestGenerator(TestGenerator):
    """
    Test generator for the Stereotyping.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Stereotyping"
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

        # Load the custom values from the test config
        custom_values = self.config.get_custom_values()
        groups = custom_values["groups"]

        # Initialize a random number generator with the seed
        random.seed(iteration_seed)
        group = [random.choice(groups) for _ in range(num_instances)]

        # Create a dictionary of sampled custom values
        sampled_values = {
            "group": group
        }

        return sampled_values

    def generate(self, model: LLM, scenario: str, custom_values: dict = {}, temperature: float = 0.0, seed: int = 42) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # Populate the templates with custom values
        control.insert("group", custom_values["group"], origin='user')
        treatment.insert("group", custom_values["group"], origin='user')

        # Populate the templates using the model and the scenario
        treatment, control = super().populate(model, treatment, control, scenario)

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
            remarks=str({"group": custom_values["group"]})
        )

        return test_case


class StereotypingMetric(RatioScaleMetric):
    """
    A metric that measures the presence and strength of Stereotyping based on a set of test results.

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results, k=np.array([1]))