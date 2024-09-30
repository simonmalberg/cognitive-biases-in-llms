from base import TestGenerator, LLM, RatioScaleMetric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np


class ReactanceTestGenerator(TestGenerator):
    """
    Test generator for Reactance.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Reactance"
        self.config: TestConfig = super().load_config(self.BIAS)

    def generate(self, model: LLM, scenario: str, custom_values: dict = {}, temperature: float = 0.0, seed: int = 42) -> TestCase:
        # Load the control and treatment templates
        variant = "reduce_behavior"
        control: Template = self.config.get_control_template(variant)
        treatment: Template = self.config.get_treatment_template(variant)

        # Populate the templates using the model and the scenario (switch control and treatment templates to make sure that the treatment template - which is the stricter one - is populated meaningfully)
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
            remarks=None
        )

        return test_case


class ReactanceMetric(RatioScaleMetric):
    """
    A metric that measures the presence and strength of Reactance based on a set of test results.

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results, k=np.array([1]))