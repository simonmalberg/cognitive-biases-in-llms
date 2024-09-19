from base import TestGenerator, LLM, Metric, MetricCalculationError
from tests import TestCase, Template, TestConfig, DecisionResult
import random
import numpy as np


class HaloEffectTestGenerator(TestGenerator):
    """
    Test generator for the Halo Effect bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Halo Effect bias.
    """

    def __init__(self):
        self.BIAS = "Halo Effect"
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

    def generate(
        self, model: LLM, scenario: str, custom_values: dict = {}, seed: int = 42
    ) -> TestCase:
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # custom values handling
        random.seed(seed)
        perception_values = custom_values["perception"]
        # Sampling one of the values
        perception = random.choice(perception_values)
        for template in [control, treatment]:
            template.insert("perception", perception, origin="user")

        # first populate the treatment template, then the control template
        treatment, control = super().populate(model, treatment, control, scenario)

        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=treatment,
            generator=model.NAME,
            scenario=scenario,
        )

        return test_case


class HaloEffectMetric(Metric):
    """
    A class that describes the quantitative evaluation of the halo effect in a model.

    Metric:
    𝔅 = k * (â₂ - â₁) / a

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    a = â₁ - â (if â₂ - â₁ < 0) or else a = ã - â₁, where ã is the maximum option, â - the minimum option (= 0).
    k is the halo type and is 1 for positive and -1 for negative.

    """

    def _compute(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        halo: np.array,
        max_option: np.array,
    ) -> np.array:
        """
        Compute the metric for the Negativity bias.

        Args:
            control_answer (np.array): The answer chosen in the control version.
            treatment_answer (np.array): The answer chosen in the treatment version.

            max_option (np.array): The maximum answer option.

        Returns:
            np.array: The metric value for the test case.
        """
        delta = treatment_answer - control_answer
        metric_value = (
            halo
            * delta
            / (
                (delta >= 0) * (max_option - control_answer)
                + (delta < 0) * control_answer
                + 10e-8
            )
        )

        return metric_value

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        try:
            # make sure all pairs are not None
            test_results = [
                pair
                for pair in test_results
                if pair[0] is not None and pair[1] is not None
            ]
            # extract the max option
            max_option = len(test_results[0][1].CONTROL_OPTIONS) - 1
            # extract answers
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
            halo = [
                [
                    insertion.text
                    for insertion in test_case.TREATMENT.get_insertions()
                    if insertion.pattern == "perception"
                ]
                for (test_case, _) in test_results
            ]
            halo = np.array([[1] if p == ["positively"] else [-1] for p in halo])
            biasedness_scores = np.mean(
                self._compute(control_answer, treatment_answer, halo, max_option)
            )
        except Exception as e:
            print(e)
            raise MetricCalculationError(f"Error computing the metric: {e}")
        return round(biasedness_scores, 2)
