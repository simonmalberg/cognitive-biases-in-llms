from base import TestGenerator, LLM, RatioScaleMetric
from tests import TestCase, Template, TestConfig, DecisionResult


class PlanningFallacyTestGenerator(TestGenerator):
    """
    Test generator for the Planning Fallacy.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Planning Fallacy"
        self.config: TestConfig = super().load_config(self.BIAS)

    def generate(
        self,
        model: LLM,
        scenario: str,
        custom_values: dict = {},
        temperature: float = 0.0,
        seed: int = 42,
    ) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # Populate the templates using the model and the scenario
        control, treatment = super().populate(
            model, control, treatment, scenario, temperature, seed
        )

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
            remarks=None,
        )

        return test_case


class PlanningFallacyMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the Planning fallacy in a model.

    Metric:
    𝔅(â₁, â₂) = (â₁ - â₂) / max(â₁, â₂) ∈ [-1, 1]

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively (control is shifted by 1: â₁ := â₁ + 1).;

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): A list of test results to be used for the metric calculation.
    """
    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        # since no shift between the cotrol and treatment indicate the planning fallacy,
        # we shift the control decision by 1 scale point to the right (if it is not already at the maximum)
        max_option = len(test_results[0][1].CONTROL_OPTIONS)
        for idx, _ in enumerate(test_results):
            test_results[idx][1].CONTROL_DECISION += 1
            test_results[idx][1].CONTROL_DECISION = min(test_results[idx][1].CONTROL_DECISION, max_option - 1)
        super().__init__(test_results)
        self.k = 1
