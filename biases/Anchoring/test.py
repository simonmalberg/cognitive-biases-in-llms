from base import TestGenerator, LLM, Metric, MetricCalculationError
from tests import TestCase, Template, TestConfig, DecisionResult
import re
import numpy as np


class AnchoringTestGenerator(TestGenerator):
    """
    Test generator for the Anchoring.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Anchoring.
    """

    def __init__(self):
        self.BIAS: str = "Anchoring"
        self.config: TestConfig = super().load_config(self.BIAS)

    def generate_all(
        self, model: LLM, scenarios: list[str], seed: int = 42
    ) -> list[TestCase]:
        # Load the custom values from the test configuration
        custom_values = self.config.get_custom_values()

        # Create test cases for all scenarios
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

    def generate(
        self, model: LLM, scenario: str, custom_values: dict = {}, seed: int = 42
    ) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # Populate the treatment template with the anchor
        np.random.seed(seed)
        # Loading the mean and max interval for the sample of numerical value
        anchor_min, anchor_max = int(custom_values["anchor"][1]), int(
            custom_values["anchor"][2]
        )
        # Loading the required distribution (should be a np.random method)
        distribution = getattr(np.random, custom_values["anchor"][0])
        # Sampling a numerical value
        anchor = str(
            int(
                distribution(
                    anchor_min,
                    anchor_max,
                )
            )
        )
        # Inserting the sample into the treatment template
        treatment.insert("anchor", anchor, origin="user")

        # Populate the templates using the model and the scenario
        control, treatment = super().populate(model, control, treatment, scenario)

        # Create a test case object
        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=treatment,
            generator=model.NAME,
            scenario=scenario,
            variant=None,
            remarks=None,
        )

        return test_case


class AnchoringMetric(Metric):
    """
    A class that describes the quantitative evaluation of the anchoring in a model.

    Metric:
    𝔅 = (‖ â₁ − a' ‖₁ − ‖ â₂ − a' ‖₁) / a ∈ [-1, 1];

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    a' is the answer option closest to the anchor value;
    a = max[‖ â₁ − a' ‖₁, ‖ â₂ − a' ‖₁].
    """

    def _compute(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        anchor: np.array,
    ) -> np.array:
        """
        Compute the metric for the anchoring.

        Args:
            control_answer (np.array): The answer chosen in the control version.
            treatment_answer (np.array): The answer chosen in the treatment version.
            anchor (np.array): The option closest to the anchor value for the test case.

        Returns:
            np.array: The metric value for the test case.
        """
        # Calculate the metric value
        delta_control = np.abs(control_answer - anchor)
        delta_treatment = np.abs(treatment_answer - anchor)
        metric_value = (delta_control - delta_treatment) / (np.maximum(
            delta_control, delta_treatment
        ) + 1e-6)

        return metric_value

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        # make sure all pairs are not None
        test_results = [
            pair for pair in test_results if pair[0] is not None and pair[1] is not None
        ]
        try:
            # extract indices of the chosen answers
            control_answer = np.array(
                [
                    [decision_result.CONTROL_DECISION]
                    for (_, decision_result) in test_results
                ]
            )
            treatment_answer = np.array(
                [
                    [decision_result.TREATMENT_DECISION]
                    for (_, decision_result) in test_results
                ]
            )
            # extract the anchor values
            anchor = [
                [
                    insertion.text
                    for insertion in test_case.TREATMENT.get_insertions()
                    if insertion.pattern == "anchor"
                ]
                for (test_case, _) in test_results
            ]
            anchor = np.array([[round(int(a[0]) / 10)] for a in anchor])
            biasedness_scores = np.mean(
                self._compute(control_answer, treatment_answer, anchor)
            )
        except Exception as e:
            print(e)
            raise MetricCalculationError(f"Error filtering test results: {e}")
        return round(biasedness_scores, 2)
