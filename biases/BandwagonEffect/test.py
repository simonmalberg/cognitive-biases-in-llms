from base import TestGenerator, LLM, Metric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np


class BandwagonEffectTestGenerator(TestGenerator):
    """
    Test generator for the Bandwagon Effect.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Bandwagon Effect"
        self.config: TestConfig = super().load_config(self.BIAS)

    def generate_all(self, model: LLM, scenarios: list[str], seed: int = 42) -> list[TestCase]:
        # Load the custom values from the test config
        config_values = self.config.get_custom_values()   # TODO: Remove this line if custom values are not needed

        # Create test cases for all scenarios
        test_cases: list[TestCase] = []
        for scenario in scenarios:
            try:
                custom_values = {
                    "custom_value": config_values["my_custom_value"][0]   # TODO: Remove this line if custom values are not needed
                }

                test_case = self.generate(model, scenario, custom_values, seed)
                test_cases.append(test_case)
            except Exception as e:
                print(f"Generating the test case failed.\nScenario: {scenario}\nSeed: {seed}")
                print(e)

        return test_cases

    def generate(self, model: LLM, scenario: str, config_values: dict = {}, seed: int = 42) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()       # TODO: Pass the variant name as a function parameter if you have more than one test variant
        treatment: Template = self.config.get_treatment_template()   # TODO: Pass the variant name as a function parameter if you have more than one test variant

        # Populate the templates with custom values
        treatment.insert_values([("my_custom_value", config_values["custom_value"])], kind='manual')   # TODO: Remove this line if custom values are not needed

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


class BandwagonEffectMetric(Metric):

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
        biasedness_scores = [self._compute(test_result) for test_result in test_results]
        return np.mean(biasedness_scores)