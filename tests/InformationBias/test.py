from core.base import TestGenerator, LLM, RatioScaleMetric
from core.testing import TestCase, DecisionResult


class InformationBiasTestGenerator(TestGenerator):
    """
    Test generator for Information Bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Information Bias.
    """

    def __init__(self):
        self.BIAS = "Information Bias"
        self.config = super().load_config(self.BIAS)

    def generate(self, model: LLM, scenario: str, custom_values: dict = {}, temperature: float = 0.0, seed: int = 42) -> TestCase:
        # Load the control and treatment template
        control = self.config.get_control_template()
        treatment = self.config.get_treatment_template()

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


class InformationBiasMetric(RatioScaleMetric):
    """
    A metric that measures the presence and strength of the Information Bias based on a set of test results.

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results, flip_treatment=True)