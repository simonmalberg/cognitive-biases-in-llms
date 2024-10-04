from base import TestGenerator, LLM, RatioScaleMetric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np
import random
import ast


class SocialDesirabilityBiasTestGenerator(TestGenerator):
    """
    Test generator for the Social Desirability Bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Social Desirability Bias"
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
        statements = custom_values["statements"]

        # Initialize a random number generator with the seed and sample a few statements
        random.seed(iteration_seed)
        statement = [random.choice(statements) for _ in range(num_instances)]
        desirable = [s[-3:] == '(T)' for s in statement]
        statement = [s[:-4] for s in statement]

        # Create a dictionary of sampled custom values
        sampled_values = {
            "statement": statement,
            "desirable": desirable
        }

        return sampled_values

    def generate(self, model: LLM, scenario: str, custom_values: dict = {}, temperature: float = 0.0, seed: int = 42) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # Insert the statement into the templates
        control.insert("statement", custom_values["statement"], origin='user')
        treatment.insert("statement", custom_values["statement"], origin='user')

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
            remarks=str({"statement": custom_values["statement"], "desirable": custom_values["desirable"]})
        )

        return test_case


class SocialDesirabilityBiasMetric(RatioScaleMetric):
    """
    A metric that measures the presence and strength of Social Desirability Bias based on a set of test results.

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results)

        # Extract from the remarks whether the statements used for the tests where socially desirable or not
        desirable = [ast.literal_eval(test_case.REMARKS)["desirable"] for (test_case, _) in self.test_results]

        self.k = np.array([[1] if d else [-1] for d in desirable])