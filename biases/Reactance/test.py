from base import TestGenerator, LLM, Metric
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
            scenario=scenario,
            control_values=None,
            treatment_values=None,
            variant=None,
            remarks=None
        )

        return test_case


class ReactanceMetric(Metric):

    def __init__(self):
        pass

    def _compute(self, test_result: tuple[TestCase, DecisionResult]) -> float:
        # Extract the decision result from the tuple
        decision_result: DecisionResult = test_result[1]

        # Extract the control and treatment decisions
        control_decision = decision_result.CONTROL_DECISION
        treatment_decision = decision_result.TREATMENT_DECISION

        # Get the number of available options in control and treatment
        SCALE_OPTIONS_CONTROL = len(decision_result.CONTROL_OPTIONS)
        SCALE_OPTIONS_TREATMENT = len(decision_result.TREATMENT_OPTIONS)

        # Calculate biasedness as the deviation between the control and treatment decision normalized by the maximum possible deviation
        if treatment_decision > control_decision:
            if SCALE_OPTIONS_TREATMENT - 1 - control_decision == 0:
                # Catch division by zero errors
                biasedness = 0.0
            else:
                biasedness = -1 * (treatment_decision - control_decision) / (SCALE_OPTIONS_TREATMENT - 1 - control_decision)
        else:
            if control_decision == 0:
                # Catch division by zero errors
                biasedness = 0.0
            else:
                biasedness = (control_decision - treatment_decision) / control_decision

        return biasedness

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        # Calculate the average biasedness score across all tests
        biasedness_scores = [self._compute(test_result) for test_result in test_results]
        return np.mean(biasedness_scores)