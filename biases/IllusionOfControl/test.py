from base import TestGenerator, LLM, RatioScaleMetric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np
import random


class IllusionOfControlTestGenerator(TestGenerator):
    """
    Test generator for the Illusion of Control.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Illusion of Control.
    """

    def __init__(self):
        self.BIAS = "Illusion of Control"
        self.config = super().load_config(self.BIAS)

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
        config_values = self.config.get_custom_values()
        treatment_variants = config_values["treatment_variants"]

        # Sample treatment variants
        random.seed(iteration_seed)
        random.shuffle(treatment_variants)
        sampled_values = {
            "treatment_variant": [treatment_variants[i % len(treatment_variants)] for i in range(num_instances)]
        }

        return sampled_values

    def generate(self, model: LLM, scenario: str, custom_values: dict = {}, temperature: float = 0.0, seed: int = 42) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # Apply the chosen treatment variant
        treatment.insert("treatment_variant", custom_values["treatment_variant"], origin='user', trim_full_stop=False)

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
            remarks=str({"treatment_variant": custom_values["treatment_variant"]})
        )

        return test_case


class IllusionOfControlMetric(RatioScaleMetric):
    """
    A metric that measures the presence and strength of the Illusion of Control based on a set of test results.

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results)