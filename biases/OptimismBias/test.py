from base import TestGenerator, LLM, Metric, MetricCalculationError
from tests import TestCase, Template, TestConfig, DecisionResult
import random
import re
import numpy as np


class OptimismBiasTestGenerator(TestGenerator):
    """
    Test generator for the Optimism bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Optimism bias.
    """

    def __init__(self):
        self.BIAS = "Optimism Bias"
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

        # Loading the possible event kind
        event_kinds = custom_values["event_kind"]
        random.seed(seed)
        # Sampling one of ['positive', 'negative']
        chosen_kind = random.choice(event_kinds)
        # Inserting the outcomes and amounts into the template
        for template in [control, treatment]:
            template.insert("event_kind", chosen_kind, origin="user")

        control, treatment = super().populate(model, control, treatment, scenario)

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


class OptimismBiasMetric(Metric):
    """
    A class that describes the quantitative evaluation of the optimism bias in a model.

    Individual metric:
    𝔅 = k * (â₂ − â₁) / ‖ â₁ − ã₁ ‖₁ ∈ [0, 1];

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    ã₁ is the farthest answer option from â₁ (either 0 or max option);
    k is the kind of event (1: positive or -1: negative).
    """

    def _compute(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        event_kind: np.array,
        max_option: np.array,
    ) -> np.array:
        """
        Compute the metric for the optimism bias.

        Args:
            control_answer (np.array): The answer from the control version.
            treatment_answer (np.array): The answer from the treatment version.
            event_kind (np.array): The kind of event (1: positive or -1: negative).
            max_option (np.array): The maximum answer option.

        Returns:
            np.array: The computed metric for the optimism bias.
        """
        delta = treatment_answer - control_answer
        metric_value = (
            event_kind
            * delta
            / (
                (delta >= 0) * (max_option - control_answer)
                + (delta < 0) * (control_answer)
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
            max_option = len(test_results[0][1].CONTROL_OPTIONS) - 1
            # extract chosen answers
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
            event_kind = np.array(
                [
                    [
                        1 if insertion.text == "positive" else -1
                        for insertion in test_case.TREATMENT.get_insertions()
                        if insertion.pattern == "event_kind"
                    ]
                    for (test_case, _) in test_results
                ]
            )
            # compute the metric and average the results
            biasedness_scores = np.mean(
                self._compute(control_answer, treatment_answer, event_kind, max_option)
            )
        except Exception as e:
            print(e)
            raise MetricCalculationError(f"Error computing the metric: {e}")
        return round(biasedness_scores, 2)
