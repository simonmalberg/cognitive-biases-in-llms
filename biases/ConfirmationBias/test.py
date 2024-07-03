from base import TestGenerator, LLM
from tests import TestCase, Template, TestConfig
import random
import numpy as np


class ConfirmationBiasTestGenerator(TestGenerator):
    """
    Test generator for the Confirmation Bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Confirmation Bias.
    """

    def __init__(self):
        self.BIAS = "Confirmation Bias"
        self.config = super().load_config(self.BIAS)

    def custom_population(self, completed_template: Template) -> None:
        """
        Custom population method for the Confirmation Bias test case.

        Args:
            completed_template (Template): The assembled template for the test case.
        """
        NUM_ARGUMENTS = 4

        # Loading the dict with custom values
        custom_values = self.config.get_custom_values()
        # Loading the possible kinds of arguments
        outcomes = custom_values['argument']
        # Sampling N pros and N cons arguments and shuffling them
        outcomes = outcomes * NUM_ARGUMENTS
        random.shuffle(outcomes)
        # insertion of the arguments into the respective answer places of the template
        to_be_filled = [f'argument_{i}' for i in range(1, 2 * NUM_ARGUMENTS + 1)]
        completed_template.insert_custom_values(to_be_filled, outcomes)

    def generate(self, model: LLM, scenario: str) -> TestCase:

        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        self.custom_population(treatment)

        treatment_inserted_values = treatment.inserted_values
        control, treatment = super().populate(model, control, treatment, scenario)

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


class ConfirmationBiasMetric(Metric):
    """
    A class that describes the quantitative evaluation of the confirmation bias in a model.

    Individual metric:
    𝔅ᵢ = max(0, [aᵢ(aᵢ⁺ − aᵢ⁻) + (1 − aᵢ)(aᵢ⁻ − aᵢ⁺)]/(aᵢ⁺ + aᵢ⁻))  ∀i = 1,.., n;


    Batch metric: [TODO: potentially can also use simple average]
    𝔅 = (𝔅₁N₁ + ... + 𝔅ₙNₙ) / (N₁ + ... + Nₙ),

    where:
    aᵢ ∈ {0,1} is the chosen answer in the control version of the i-th test;
    aᵢ⁺ is the number of pro-arguments selected in the treatment version of the i-th test;
    aᵢ⁻ is the number of con-arguments selected in the treatment version of the i-th test;
    Nᵢ is the number of arguments in the treatment version of the i-th test;
    n is number of test cases in the batch.

    Attributes:
        overall (bool): A flag that is used to indicate that a single result per batch of test is required.
    """

    def __init__(self, overall: bool):
        self.overall = overall

    def compute_internal(
        self,
        answer: np.array,
        pro_answer: np.array,
        con_answer: np.array,
        n_args: np.array,
    ) -> np.array:
        """
        Computes the confirmation bias metric for the given batch of test instances.

        Args:
            answer (np.array, shape (batch, 1)): The answer(s) chosen in the control version(s).
            pro_answer (np.array, shape (batch, 1)): The number of pro-arguments chosen in the treatment version(s).
            con_answer (np.array, shape (batch, 1)): The number of con-arguments chosen in the treatment version(s).
            n_args (np.array, shape (batch, 1)): The number of arguments available in the treatment version(s).

        Returns:
            The confirmation bias metric value.
        """
        result = np.maximum(
            0,
            (
                answer * (pro_answer - con_answer)
                + (1 - answer) * (con_answer - pro_answer)
            )
            / (pro_answer + con_answer),
        )
        if not self.overall:
            return result

        result = np.sum(result * n_args) / np.sum(n_args)

        return result

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        pass