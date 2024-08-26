from base import TestGenerator, LLM, Metric, MetricCalculationError
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np


class AnthropomorphismTestGenerator(TestGenerator):
    """
    Test generator for the Anthropomorphism.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Anthropomorphism"
        self.config: TestConfig = super().load_config(self.BIAS)

    def generate_all(
        self, model: LLM, scenarios: list[str], seed: int = 42
    ) -> list[TestCase]:
        # Create test cases for all scenarios
        test_cases: list[TestCase] = []
        for scenario in scenarios:
            try:
                test_case = self.generate(model, scenario, None, seed)
                test_cases.append(test_case)
            except Exception as e:
                print(
                    f"Generating the test case failed.\nScenario: {scenario}\nSeed: {seed}"
                )
                print(e)

        return test_cases

    def generate(
        self, model: LLM, scenario: str, custom_values: dict = {}, seed: int = 42
    ) -> TestCase:
        # Load the treatment templates
        treatment: Template = self.config.get_treatment_template()
        # Populate the template using the model and the scenario
        _, _ = super().populate(model, None, treatment, scenario)

        # Create a test case object
        test_case = TestCase(
            bias=self.BIAS,
            control=None,
            treatment=treatment,
            generator=model.NAME,
            scenario=scenario,
            control_values=None,
            treatment_values=None,
            variant=None,
            remarks=None,
        )

        return test_case


class AnthropomorphismMetric(Metric):
    """
    A class that describes the quantitative evaluation of the Anthropomorphism in a model.

    Metric:
    𝔅 = I[â ∈ A] ∈ {0, 1}
    where:
    â is the chosen answer for the treatment version, A is the set of answers that are anthropomorphic (0 and 1 in the current test).
    """

    def _compute(self, treatment_answer: np.array) -> np.array:
        """
        Compute the metric for the Anthropomorphism.

        Args:
            control_answer (np.array): The answer chosen in the control version.
            treatment_answer (np.array): The answer chosen in the treatment version.

        Returns:
            np.array: The metric value for the test case.
        """
        metric_value = treatment_answer == 0 or treatment_answer == 1

        return metric_value

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        try:
            # make sure all pairs are not None
            test_results = [
                pair
                for pair in test_results
                if pair[0] is not None and pair[1] is not None
            ]
            # extract chosen answers
            treatment_answer = np.array(
                [
                    decision_result.TREATMENT_DECISION
                    for (_, decision_result) in test_results
                ]
            )
            # compute the biasedness scores
            biasedness_scores = np.mean(self._compute(treatment_answer))
        except Exception as e:
            print(e)
            raise MetricCalculationError(f"Error filtering test results: {e}")
        return biasedness_scores
