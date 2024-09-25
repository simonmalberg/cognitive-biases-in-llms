from base import TestGenerator, LLM, Metric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np


class InformationBiasTestGenerator(TestGenerator):
    """
    Test generator for Information Bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Information Bias.
    """

    def __init__(self):
        self.BIAS = "InformationBias"
        self.config = super().load_config(self.BIAS)

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

    def generate(self, model: LLM, scenario: str, config_values: dict = {}, seed: int = 42) -> TestCase:

        # Load the treatment template
        treatment: Template = self.config.get_treatment_template()

        # Sample prior confidence in course of action
        min, max, step = config_values["prior"]
        prior = np.random.choice(np.arange(int(min), int(max)+1, int(step)), size=1)[0]

        # Sample posteriors lower and higher than prior
        min, max, step = config_values["posterior_high"]
        posterior_high = np.random.choice(np.arange(int(prior), int(max)+1, int(step)), size=1)[0]
        min, max, step = config_values["posterior_low"]
        posterior_low = np.random.choice(np.arange(int(min), int(prior), int(step)), size=1)[0]

        # Insert the sampled values into the treatment template
        treatment.insert('prior', str(prior)+"%", origin='user')
        treatment.insert('posterior_high', str(posterior_high)+"%", origin='user')
        treatment.insert('posterior_low', str(posterior_low)+"%", origin='user')

        # Populate the templates using the model and the scenario
        _, treatment = super().populate(model, None, treatment, scenario)

        # Create a test case object
        test_case = TestCase(
            bias=self.BIAS,
            control=None,
            treatment=treatment,
            generator=model.NAME,
            variant=None,
            scenario=scenario
        )

        return test_case


class InformationBiasMetric(Metric):

    def __init__(self):
        pass

    def _compute(self, test_result: tuple[TestCase, DecisionResult]) -> float:
        pass

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        pass