from base import TestGenerator, LLM, Metric
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

    def _custom_population(self, completed_template: Template) -> str:
        """
        Custom population method for the Halo Effect test case.

        Args:
            completed_template (Template): The assembled template for the test case.

        Returns:
            experience_sentiment (str): The sampled custom value used in both templates.
        """
        # Loading the dict with custom values
        custom_values = self.config.get_custom_values()
        experience_sentiments = custom_values['experience_sentiment']
        # Sampling one of the outcomes
        experience_sentiment = random.choice(experience_sentiments)
        completed_template.insert_custom_values(['experience_sentiment'], [experience_sentiment])

        return experience_sentiment

    def generate_all(self, model: LLM, scenarios: list[str], config_values: dict = {}, seed: int = 42) -> list[TestCase]:
        # TODO Implement functionality to generate multiple test cases at once (potentially following the ranges or distributions outlined in the config values)
        pass

    def generate(self, model: LLM, scenario: str, config_values: dict = {}, seed: int = 42) -> TestCase:
        # TODO Refactor to use only the config values passed to this method (i.e., only the values to be applied to the generation of this very test case)
        
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # sample a sentiment for control version, insert it in the treatment
        experience_sentiment = self._custom_population(control)
        treatment.insert_custom_values(['experience_sentiment'], [experience_sentiment])
        # get dictionary of inserted values
        control_inserted_values = control.inserted_values
        treatment_inserted_values = treatment.inserted_values

        control, treatment = super().populate(model, control, treatment, scenario)

        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=treatment,
            generator=model.NAME,
            control_custom_values=control_inserted_values,
            treatment_custom_values=treatment_inserted_values,
            scenario=scenario
        )

        return test_case


class HaloEffectMetric(Metric):
    """
    A class that describes the quantitative evaluation of the halo effect bias in a model.

    Individual metric:
    𝔅 = ‖ â₂ − â₁ ‖₁ / ‖ â₁ − ã₁ ‖₁ ∈ [0, 1];

    Batch metric:
    𝔅 = (𝔅₁ + ... + 𝔅ₙ) / n,

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    ã₁ is the farthest answer option from â₁;
    n is number of test cases in the batch.

    Attributes:
        overall (bool): A flag that is used to indicate that a single result per batch of test is required.

    """

    def __init__(self, overall: bool):
        self.overall = overall

    def _compute(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        answer_options: np.array,
    ) -> np.array:
        """
        Computes the halo effect bias metric for the given batch of test instances.

        Args:
            control_answer (np.array, shape (batch, 1)): The answer(s) chosen for the control variant.
            treatment_answer (np.array, shape (batch, 1)): The answer(s) chosen for the treatment variant.
            answer_options (np.array, shape (batch, num_options)): The answer options available in the test case.

        Returns:
            The halo effect bias metric value.
        """
        farthest_idx = np.argmax(
            np.abs(answer_options - control_answer), axis=1, keepdims=True
        )
        farthest_val = np.take_along_axis(answer_options, farthest_idx, axis=1)
        result = np.abs(control_answer - treatment_answer) / np.abs(
            control_answer - farthest_val + 1e-8
        )
        if self.overall:
            return np.mean(result)

        return result

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        # TODO Implement computation of this metric
        pass