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
        preference_values = custom_values["preference"]
        # Sampling one of the values
        perception = random.choice(perception_values)
        preferential_brand = random.choice(preference_values)
        other_brand = (
            preference_values[0]
            if preferential_brand == preference_values[1]
            else preference_values[1]
        )
        perceived_brand = (
            other_brand if perception == "positive" else preferential_brand
        )
        for template in [control, treatment]:
            template.insert("perception", perception, origin="user")
            template.insert("preference", preferential_brand, origin="user")
            template.insert("perceived_brand", perceived_brand, origin="user")

        control, treatment = super().populate(model, control, treatment, scenario)

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
    A class that describes the quantitative evaluation of the halo effect bias in a model.

    Metric:
    𝔅 = I[â₂ != â₁] * (1 - 2I[â₁ != p])

    where:
    I is the indicator function;
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively.
    p is the advantageous product (0 or 1).

    """

    def _compute(
        self, control_answer: np.array, treatment_answer: np.array, preference: np.array
    ) -> np.array:
        """
        Computes the halo effect bias metric for the given batch of test instances.

        Args:
            control_answer (np.array, shape (batch, 1)): The answer(s) chosen for the control variant.
            treatment_answer (np.array, shape (batch, 1)): The answer(s) chosen for the treatment variant.
            preference (np.array, shape (batch, 1)): The advantageous product (0 or 1).

        Returns:
            The halo effect bias metric value.
        """
        metric_values = (control_answer != treatment_answer) * (
            1 - 2 * (control_answer != preference)
        )

        return metric_values

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        try:
            # make sure all pairs are not None
            test_results = [
                pair
                for pair in test_results
                if pair[0] is not None and pair[1] is not None
            ]
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
            preference = [
                [
                    insertion.text
                    for insertion in test_case.CONTROL.get_insertions()
                    if insertion.pattern == "preference"
                ]
                for (test_case, _) in test_results
            ]
            print(preference)
            preference = np.array([[0] if p == ["A"] else [1] for p in preference])
            print(preference)
            biasedness_scores = np.mean(
                self._compute(control_answer, treatment_answer, preference)
            )
        except Exception as e:
            print(e)
            raise MetricCalculationError(f"Error filtering test results: {e}")
        return biasedness_scores
