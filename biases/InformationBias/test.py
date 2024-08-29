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

    def generate_all(self, model: LLM, scenarios: list[str], seed: int = 42) -> list[TestCase]:

        # Get a list of all variants in the test config
        variants = self.config.get_variants()

        # Create test cases for all variants and scenarios
        test_cases: list[TestCase] = []
        for variant in variants:
            for scenario in scenarios:
                try:
                    instance_values = {
                        "variant": variant
                    }

                    test_case = self.generate(model, scenario, instance_values, seed)
                    test_cases.append(test_case)
                except Exception as e:
                    print("Generating the test case failed.")
                    print(f"Variant: {variant}")
                    print(f"Scenario: {scenario}")
                    print(f"Seed: {seed}")
                    print(e)

        return test_cases

    def generate(self, model: LLM, scenario: str, config_values: dict = {}, seed: int = 42) -> TestCase:
        # Load the treatment template for the selected test variant
        variant = config_values["variant"]
        treatment: Template = self.config.get_treatment_template(variant)

        # Populate the templates using the model and the scenario
        _, treatment = super().populate(model, None, treatment, scenario)

        # Create a test case object
        test_case = TestCase(
            bias=self.BIAS,
            control=None,
            treatment=treatment,
            generator=model.NAME,
            variant=variant,
            scenario=scenario
        )

        return test_case


class InformationBiasMetric(Metric):

    def __init__(self):
        pass

    def _compute(self, test_result: tuple[TestCase, DecisionResult]) -> float:
        # Extract the decision result from the tuple
        decision_result: DecisionResult = test_result[1]

        # Extract the chosen option
        treatment_decision = int(decision_result.TREATMENT_DECISION)

        # Calculate the biasedness
        return treatment_decision

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        biasedness_scores = [self._compute(test_result) for test_result in test_results]
        return np.around(biasedness_scores, 2)