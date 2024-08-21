from base import TestGenerator, LLM, Metric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np
import random

class FramingEffectTestGenerator(TestGenerator):
    """
    Test generator for the Framing Effect.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Framing Effect"
        self.config: TestConfig = super().load_config(self.BIAS)

    def generate_all(
        self, model: LLM, scenarios: list[str], seed: int = 42
    ) -> list[TestCase]:
        # Load the custom values from the test config
        custom_values = self.config.get_custom_values()

        # Create test cases for all provided scenarios
        test_cases: list[TestCase] = []
        for scenario in scenarios:
            try:
                test_case = self.generate(model, scenario, custom_values, seed)
                test_cases.append(test_case)
            except Exception as e:
                print(
                    f"Generating the test case failed.\nScenario: {scenario}\nSeed: {seed}"
                )
                print(e)

        return test_cases

    def generate(self, model: LLM, scenario: str, custom_values: dict = {}, seed: int = 42) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # Populate the templates with custom values
        # Loading the required distribution (should be a np.random method)
        first_percentage = custom_values["first_percentage"]
        distribution = getattr(np.random, first_percentage[0])
        first_percentage = distribution(
            float(first_percentage[1]), float(first_percentage[2])
        )
        control.insert("first_percentage", str(first_percentage))
        treatment.insert("second_percentage", str(100-first_percentage))
        # Get dictionary of inserted values
        control_values = control.inserted_values
        treatment_values = treatment.inserted_values

        # Populate the templates using the model and the scenario
        control, treatment = super().populate(model, control, treatment, scenario)

        # Create a test case object
        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=treatment,
            generator=model.NAME,
            scenario=scenario,
            control_values=control_values,
            treatment_values=treatment_values,
            variant=None,
            remarks=None
        )

        return test_case


class FramingEffectMetric(Metric):

    def __init__(self):
        pass

    def _compute(self, test_result: tuple[TestCase, DecisionResult]) -> float:
        # Extract the test case and decision result from the tuple
        test_case: TestCase = test_result[0]
        decision_result: DecisionResult = test_result[1]

        # Calculate the biasedness
        biasedness = 0.0   # TODO: Implement calculation of biasedness here

        return biasedness

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        # Calculate the average biasedness score across all tests
        return 0
        biasedness_scores = [self._compute(test_result) for test_result in test_results]
        return np.mean(biasedness_scores)