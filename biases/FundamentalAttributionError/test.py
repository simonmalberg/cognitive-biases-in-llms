from base import TestGenerator, LLM, Metric, MetricCalculationError
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np


class FundamentalAttributionErrorTestGenerator(TestGenerator):
    """
    Test generator for the Fundamental Attribution Error.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Fundamental Attribution Error"
        self.config: TestConfig = super().load_config(self.BIAS)

    def generate_all(
        self, model: LLM, scenarios: list[str], seed: int = 42
    ) -> list[TestCase]:
        # Load the custom values from the test config
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
        
        # Populate the templates with custom values
        np.random.seed(seed)
        reason_control, reason_treatment = (
            custom_values['reason_control'],
            custom_values['reason_treatment']
        )
        # Sampling the order of the reasons
        index = np.random.choice(range(len(reason_control)))
        for template in [control, treatment]:
            template.insert("control_reason", reason_control[index], origin="user")
            template.insert("other_control_reason", reason_control[1 - index], origin="user")
            template.insert("treatment_reason", reason_treatment[index], origin="user")
            template.insert("other_treatment_reason", reason_treatment[1 - index], origin="user")

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
            remarks=None,
        )

        return test_case


class FundamentalAttributionErrorMetric(Metric):
    """
    A class that describes the quantitative evaluation of the FAE in a model.

    Metric:
    𝔅 = â₁ mod 2 - â₂ mod 2 ∈ [-1, 1]
    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively (0 for dispositional, 1 for situational);
    """

    def _compute(
        self, control_answer: np.array, treatment_answer: np.array
    ) -> np.array:
        """
        Compute the metric for the Confirmation bias.

        Args:
            control_answer (np.array): The answer chosen in the control version.
            treatment_answer (np.array): The answer chosen in the treatment version.

        Returns:
            np.array: The metric value for the test case.
        """
        metric_value = control_answer % 2 - treatment_answer % 2

        return metric_value

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        try:
            # make sure all pairs are not None
            test_results = [
                pair
                for pair in test_results
                if pair[0] is not None and pair[1] is not None
            ]
            # extract strings of the chosen answers
            control_answer = np.array([
                [decision_result.CONTROL_DECISION]
                for (_, decision_result) in test_results
            ])
            treatment_answer = np.array([
                [decision_result.TREATMENT_DECISION]
                for (_, decision_result) in test_results
            ])
            # compute the biasedness scores
            biasedness_scores = np.mean(self._compute(control_answer, treatment_answer))
        except Exception as e:
            print(e)
            raise MetricCalculationError(f"Error computing the metric: {e}")
        return biasedness_scores
