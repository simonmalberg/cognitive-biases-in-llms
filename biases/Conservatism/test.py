from base import TestGenerator, LLM, Metric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np


class ConservatismTestGenerator(TestGenerator):
    """
    Test generator for the Conservatism.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Conservatism"
        self.config: TestConfig = super().load_config(self.BIAS)

    def generate_all(self, model: LLM, scenarios: list[str], seed: int = 42) -> list[TestCase]:
        # Create test cases for all scenarios
        test_cases: list[TestCase] = []
        for scenario in scenarios:
            try:
                test_case = self.generate(model, scenario, {}, seed)
                test_cases.append(test_case)
            except Exception as e:
                print(f"Generating the test case failed.\nScenario: {scenario}\nSeed: {seed}")
                print(e)

        return test_cases

    def generate(self, model: LLM, scenario: str, config_values: dict = {}, seed: int = 42) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # Populate the templates using the model and the scenario
        control, treatment = super().populate(model, control, treatment, scenario)

        # Create a test case object
        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=treatment,
            generator=model.NAME,
            scenario=scenario,
            control_values=None,
            treatment_values=None,
            variant=None,
            remarks=None
        )

        return test_case


class ConservatismMetric(Metric):

    def __init__(self):
        pass

    def _compute(self, test_result: tuple[TestCase, DecisionResult]) -> float:
        # Extract the decision result from the tuple
        decision_result: DecisionResult = test_result[1]

        # Extract the control and treatment decisions
        control_preference = decision_result.CONTROL_DECISION
        treatment_preference = decision_result.TREATMENT_DECISION

        # Get the number of available options to calculate the center and diameter of the scale for control
        SCALE_OPTIONS_CONTROL = len(decision_result.CONTROL_OPTIONS)
        SCALE_DIAMETER_CONTROL = SCALE_OPTIONS_CONTROL // 2
        SCALE_CENTER_CONTROL = SCALE_DIAMETER_CONTROL + 1

        # Do the same for treatment
        SCALE_OPTIONS_TREATMENT = len(decision_result.TREATMENT_OPTIONS)
        SCALE_DIAMETER_TREATMENT = SCALE_OPTIONS_TREATMENT // 2
        SCALE_CENTER_TREATMENT = SCALE_DIAMETER_TREATMENT + 1

        # Calculate the normalized biasedness for both, the control and the treatment decision
        bias_control = -1 * (control_preference - SCALE_CENTER_CONTROL) / SCALE_DIAMETER_CONTROL
        bias_treatment = -1 * (treatment_preference - SCALE_CENTER_TREATMENT) / SCALE_DIAMETER_TREATMENT

        # Calculate the average biasenedness of the control and the treatment decision
        biasedness = (bias_control + bias_treatment) / 2

        return biasedness

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        # Calculate the average biasedness score across all tests
        biasedness_scores = [self._compute(test_result) for test_result in test_results]
        return np.mean(biasedness_scores)