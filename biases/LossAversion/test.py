from base import TestGenerator, LLM, Metric
from tests import TestCase, Template, TestConfig, DecisionResult
import random
import numpy as np


class LossAversionTestGenerator(TestGenerator):
    """
    Test generator for the Loss Aversion bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Loss Aversion bias.
    """

    def __init__(self):
        self.BIAS = "Loss Aversion"
        self.config = super().load_config(self.BIAS)

    def _custom_population(self, completed_template: Template) -> None:
        """
        Custom population method for the Loss Aversion test case.

        Args:
            completed_template (Template): The assembled template for the test case.
        """
        # Loading the dict with custom values
        custom_values = self.config.get_custom_values()
        # Loading the possible outcomes and amounts
        outcome = custom_values['outcome']
        amount = custom_values['amount']

        # Sampling one of ['gain', 'loss'] and taking the index:
        first_outcome = random.choice(outcome)
        first_idx = outcome.index(first_outcome)
        # Taking the other outcome as the second one:
        second_outcome = outcome[(first_idx + 1) % 2]
        # Taking the respective amounts
        first_amount = amount[first_idx]
        second_amount = amount[(first_idx + 1) % 2]

        # Inserting the outcomes and amounts into the template
        patterns = ['first_outcome', 'second_outcome', 'first_amount', 'second_amount']
        values = [first_outcome, second_outcome, first_amount, second_amount]
        completed_template.insert_custom_values(patterns, values)

        # Sampling the value of lambda - TODO: might be better to sample a vector for several tests, discuss it
        lambda_coef = round(random.uniform(1, 2), 1) # TODO: select the distribution
        completed_template.insert_custom_values(['lambda_coef'], [str(lambda_coef)])

    def generate_all(self, model: LLM, scenarios: list[str], config_values: dict = {}, seed: int = 42) -> list[TestCase]:
        # TODO Implement functionality to generate multiple test cases at once (potentially following the ranges or distributions outlined in the config values)
        pass        

    def generate(self, model: LLM, scenario: str, config_values: dict = {}, seed: int = 42) -> TestCase:
        # TODO Refactor to use only the config values passed to this method (i.e., only the values to be applied to the generation of this very test case)

        treatment: Template = self.config.get_treatment_template()
        self._custom_population(treatment)
        treatment_custom_values = treatment.inserted_values

        _, treatment = super().populate(model, None, treatment, scenario)

        # Create a test case object and remember the sampled lambda value
        test_case = TestCase(
            bias=self.BIAS,
            control=None,
            treatment=treatment,
            generator=model.NAME,
            control_custom_values=None,
            treatment_custom_values=treatment_custom_values,
            scenario=scenario
        )

        return test_case


class LossAversionMetric(Metric):
    """
    A class that describes the quantitative evaluation of the loss aversion bias in a model.

    Individual metric:
    𝔅ᵢ = aᵢ ∀i = 1,.., n;


    Batch metric:
    𝔅 = 1 - (𝔅₁/λ₁ + ... + 𝔅ₙ/λₙ) / (1/λ₁ + ... + 1/λₙ) ∈ [0, 1],

    where:
    aᵢ ∈ {0,1} is the chosen answer for the i-th test;
    λᵢ is the loss aversion hyperparameter in the i-th test, decreased by 1 (for sharpness purpose).

    Attributes:
        overall (bool): A flag that is used to indicate that a single result per batch of test is required.
    """

    def __init__(self, overall: bool):
        self.overall = overall

    def _compute(self, answer: np.array, lambda_val: np.array) -> np.array:
        """
        Computes the loss aversion bias metric for the given batch of test instances.

        Args:
            answer (np.array, shape (batch, 1)): The answer(s) chosen.
            lambda_val (np.array, shape (batch, 1)): The loss aversion hyperparameter(s).

        Returns:
            The loss aversion bias metric value.
        """
        if not self.overall:
            return answer

        lambda_val = lambda_val
        result = 1 - np.sum(answer / lambda_val) / np.sum(1 / lambda_val)

        return result

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        # TODO Implement computation of this metric
        pass