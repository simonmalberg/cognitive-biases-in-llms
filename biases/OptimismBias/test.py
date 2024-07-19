from base import TestGenerator, LLM, Metric, MetricCalculationError
from tests import TestCase, Template, TestConfig, DecisionResult
import random
import re
import numpy as np


class OptimismBiasTestGenerator(TestGenerator):
    """
    Test generator for the Optimism Bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Optimism Bias.
    """

    def __init__(self):
        self.BIAS = "Optimism Bias"
        self.config = super().load_config(self.BIAS)

    def _custom_population(self, completed_template: Template) -> None:
        """
        Custom population method for the Optimism Bias test case.

        Args:
            completed_template (Template): The assembled template for the test case.
        """
        # Loading the dict with custom values
        custom_values = self.config.get_custom_values()
        # Loading the possible event kind
        event_kinds = custom_values["event_kind"]

        # Sampling one of ['positive', 'negative']
        chosen_kind = random.choice(event_kinds)

        # Inserting the outcomes and amounts into the template
        patterns, values = ["event_kind"], [chosen_kind]
        completed_template.insert_values(list(zip(patterns, values)), kind="manual")

        return chosen_kind

    def generate_all(
        self, model: LLM, scenarios: list[str], config_values: dict = {}, seed: int = 42
    ) -> list[TestCase]:
        pass

    def generate(
        self, model: LLM, scenario: str, config_values: dict = {}, seed: int = 42
    ) -> TestCase:

        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # sample a sentiment for control version, insert it in the treatment
        chosen_kind = self._custom_population(control)
        treatment.insert_values(list(zip(["event_kind"], [chosen_kind])), kind="manual")
        # get dictionary of inserted values
        control_values = control.inserted_values
        treatment_values = treatment.inserted_values

        control, treatment = super().populate(model, control, treatment, scenario)

        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=treatment,
            generator=model.NAME,
            control_values=control_values,
            treatment_values=treatment_values,
            scenario=scenario,
        )

        return test_case


class OptimismBiasMetric(Metric):
    """
    A class that describes the quantitative evaluation of the optimism bias in a model.

    Individual metric:
    𝔅 = ‖ â₂ − â₁ ‖₁ / ‖ â₁ − ã₁ ‖₁ ∈ [0, 1];

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    ã₁ is the farthest answer option from â₁.
    """

    def _compute(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        answer_options: np.array,
    ) -> np.array:
        """
        Compute the metric for the optimism bias.

        Args:
            control_answer (np.array): The answer from the control version.
            treatment_answer (np.array): The answer from the treatment version.
            answer_options (np.array): All answer options.

        Returns:
            np.array: The computed metric for the optimism bias.
        """
        farthest_idx = np.argmax(
            np.abs(answer_options - control_answer), axis=1, keepdims=True
        )
        farthest_val = np.take_along_axis(answer_options, farthest_idx, axis=1)
        result = np.abs(control_answer - treatment_answer) / np.abs(
            control_answer - farthest_val + 1e-8
        )

        return result

    # TODO: consider moving this method to the base class if it is used in multiple metrics
    def assemble_options(self, options_list: list[dict]) -> np.array:
        """
        Assemble the answer options into a numpy array.

        Args:
            options (dict): The answer options for the test case.

        Returns:
            np.array: The assembled answer options array.
        """
        answer_options = np.array([])
        for options in options_list:
            numerical_options = [
                int(re.findall(r"\b\d+\b", s)[0]) for s in options.values()
            ]
            if not answer_options.size:
                answer_options = np.array([numerical_options])
            else:
                answer_options = np.vstack((answer_options, numerical_options))

        return answer_options

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:

        try:
            # in this metric, we don't use the test case, only the decision results
            # extract the answer options (identical for treatment and control for this bias)
            answer_options = self.assemble_options(
                [
                    decision_result.CONTROL_OPTIONS
                    for (_, decision_result) in test_results
                ]
            )
            # extract indices of the chosen answers (-1 because the option indices are 1-indexed)
            control_answer_idx = (
                np.array(
                    [
                        decision_result.CONTROL_DECISION
                        for (_, decision_result) in test_results
                    ]
                )
                - 1
            )
            treatment_answer_idx = (
                np.array(
                    [
                        decision_result.TREATMENT_DECISION
                        for (_, decision_result) in test_results
                    ]
                )
                - 1
            )
            # extract the chosen answers (-1 because the option indices are 1-indexed)
            control_answer = np.take_along_axis(
                answer_options, control_answer_idx, axis=1
            )
            treatment_answer = np.take_along_axis(
                answer_options, treatment_answer_idx, axis=1
            )
            # compute the metric
            _result = self._compute(control_answer, treatment_answer, answer_options)
            # average the results (necessary if there are multiple test results)
            result = np.mean(_result)
        except Exception as e:
            raise MetricCalculationError("The metric could not be computed.")

        return result
