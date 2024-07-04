from base import TestGenerator, LLM, Metric
from tests import TestCase, Template, TestConfig, DecisionResult
import re
import numpy as np


# TODO: add functionality to generate the anchor from the answer options' interval
class AnchoringBiasTestGenerator(TestGenerator):
    """
    Test generator for the Anchoring Bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Anchoring Bias.
    """

    def __init__(self):
        self.BIAS = "Anchoring Bias"
        self.config = super().load_config(self.BIAS)

    def _custom_population(self, model: LLM, completed_template: Template) -> None:
        """
        Custom population method for the Anchoring Bias test case.

        Args:
            model (LLM): The LLM model to use for generating the anchor sentence.
            completed_template (Template): The assembled template with scenario for the test case.
        """
        custom_values = self.config.get_custom_values()
        # Loading the anchor sentence generation prompt
        anchor_sentence = custom_values['anchor_sentence'][0]
        # generate the anchor sentence
        anchor_sentence = model.generate_misc(anchor_sentence) # TODO This line causes an error when used with RandomModel as it doesn't have a generate_misc function
        # Inserting the anchor into the template
        completed_template.insert_custom_values(['anchor_sentence'], [anchor_sentence])
        # Explicitly extract the numerical value from the generated anchor sentence
        anchor = re.findall(r'\d+', anchor_sentence)
        assert len(anchor) == 1, "The anchor sentence should contain exactly one numerical value"
        # technically, we don't insert anything (just remember the value in template)
        completed_template.insert_custom_values(['anchor'], anchor)

    def generate_all(self, model: LLM, scenarios: list[str], config_values: dict = {}, seed: int = 42) -> list[TestCase]:
        # TODO Implement functionality to generate multiple test cases at once (potentially following the ranges or distributions outlined in the config values)
        pass

    def generate(self, model: LLM, scenario: str, config_values: dict = {}, seed: int = 42) -> TestCase:
        # TODO Refactor to use only the config values passed to this method (i.e., only the values to be applied to the generation of this very test case)

        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # Insert custom anchor sentence and remember the anchor value
        self._custom_population(model, treatment)
        treatment_inserted_values = treatment.inserted_values

        control, treatment = super().populate(model, control, treatment, scenario)

        # Create a test case object
        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=treatment,
            generator=model.NAME,
            control_custom_values=None,
            treatment_custom_values=treatment_inserted_values,
            scenario=scenario
        )

        return test_case


class AnchoringBiasMetric(Metric):
    """
    A class that describes the quantitative evaluation of the anchoring bias in a model.
    Currently, two variations of individual metric are implemented: anchor agnostic and anchor specific,

    Individual agnostic:
    𝔅 = ‖ â₂ − â₁ ‖₁ / ‖ â₁ − ã₁ ‖₁ ∈ [0, 1];

    Individual specific:
    𝔅' = max[0, (‖ â₁ − a' ‖₁ − ‖ â₂ − a' ‖₁) / ‖ â₁ − a' ‖₁] ∈ [0, 1];

    Batch metric:
    𝔅 = (𝔅₁ + ... + 𝔅ₙ) / n,

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    ã₁ is the farthest answer option from â₁;
    a' is the answer option closest to the anchor value;
    n is number of test cases in the batch.

    Attributes:
        anchor_agnostic (bool): A flag that is used for choice between two variations of the metric.
        overall (bool): A flag that is used to indicate that a single result per batch of test is required.

    """

    def __init__(self, anchor_agnostic: bool, overall: bool):
        self.anchor_agnostic = anchor_agnostic
        self.overall = overall

    def anchor_agnostic_metric(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        answer_options: np.array,
    ) -> np.array:
        """
        Computes the anchor agnostic anchoring bias metric for the given batch of test instances.

        Args:
            control_answer (np.array, shape (batch, 1)): The answer(s) chosen for the control variant.
            treatment_answer (np.array, shape (batch, 1)): The answer(s) chosen for the treatment variant.
            answer_options (np.array, shape (batch, num_options)): The answer options available in the test case.

        Returns:
            The anchor agnostic anchoring bias metric value.
        """
        farthest_idx = np.argmax(
            np.abs(answer_options - control_answer), axis=1, keepdims=True
        )
        farthest_val = np.take_along_axis(answer_options, farthest_idx, axis=1)
        result = np.abs(control_answer - treatment_answer) / np.abs(
            control_answer - farthest_val + 1e-8
        )

        return result

    def anchor_specific_metric(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        answer_options: np.array,
        anchor: np.array,
    ) -> np.array:
        """
        Computes the anchor specific anchoring bias metric for the given batch of test instances.

        Args:
            control_answer (np.array, shape (batch, 1)): The answer(s) chosen for the control variant.
            treatment_answer (np.array, shape (batch, 1)): The answer(s) chosen for the treatment variant.
            answer_options (np.array, shape (batch, num_options)): The answer options available in the test case.
            anchor (np.array, shape (batch, 1)): The anchor value(s) used in the test case.

        Returns:
            The anchor specific anchoring bias metric value.
        """
        closest_idx = np.argmin(np.abs(anchor - answer_options), axis=1, keepdims=True)
        closest_val = np.take_along_axis(answer_options, closest_idx, axis=1)
        result = (
            np.abs(control_answer - closest_val)
            - np.abs(treatment_answer - closest_val)
        ) / np.abs(control_answer - closest_val + 1e-8)
        result = np.maximum(0, result)

        return result

    def _compute(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        answer_options: np.array,
        anchor: np.array,
    ) -> np.array:
        """
        Computes the chosen anchoring bias metric.

        Args:
            control_answer (np.array, shape (batch, 1)): The answer(s) chosen for the control variant.
            treatment_answer (np.array, shape (batch, 1)): The answer(s) chosen for the treatment variant.
            answer_options (np.array, shape (batch, num_options)): The answer options available in the test case.
            anchor (np.array, shape (batch, 1)): The anchor value(s) used in the test case.

        Returns:
            The anchoring bias metric value.
        """

        if self.anchor_agnostic:
            if self.overall:
                return np.mean(
                    self.anchor_agnostic_metric(
                        control_answer, treatment_answer, answer_options
                    )
                )
            return self.anchor_agnostic_metric(
                control_answer, treatment_answer, answer_options
            )
        if self.overall:
            return np.mean(
                self.anchor_specific_metric(
                    control_answer, treatment_answer, answer_options, anchor
                )
            )
        return self.anchor_specific_metric(
            control_answer, treatment_answer, answer_options, anchor
        )

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        # TODO Implement computation of this metric
        pass