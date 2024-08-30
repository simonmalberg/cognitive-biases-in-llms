from base import TestGenerator, LLM, Metric, MetricCalculationError
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np
import random


class BandwagonEffectTestGenerator(TestGenerator):
    """
    Test generator for the Bandwagon Effect.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Bandwagon Effect"
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
        majority_opinions = custom_values["majority_opinion"]
        random.seed(seed)
        # Sampling one of ['A', 'B']
        majority_opinion = random.choice(majority_opinions)
        # Inserting the sample into the template
        for template in [control, treatment]:
            template.insert("majority_opinion", majority_opinion, origin="user")
        # Get dictionary of inserted values
        control_values = control.inserted_values
        treatment_values = treatment.inserted_values

        # Populate the templates using the model and the scenario
        control, treatment = super().populate(model, control, treatment, scenario)

        # Create a test case object
        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=treatment,
            generator=model.NAME,
            scenario=scenario,
            control_values=control_values,
            treatment_values=treatment_values,
            variant=None,
            remarks=None,
        )

        return test_case


class BandwagonEffectMetric(Metric):
    """
    Metric calculator for the Bandwagon Effect.

    Metric:
    𝔅 = ∑ I{â₁ = â₂ ∧ â₁ = a} - ∑ I{â₁ = â₂ ∧ â₁ != a}

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    a is the majority opinion inserted in the test case.

    """

    def _compute(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        majority_opinion: np.array,
    ) -> np.array:
        """
        Compute the metric for the Bandwagon effect.

        Args:
            control_answer (np.array): The answer chosen in the control version.
            treatment_answer (np.array): The answer chosen in the treatment version.
            majority_opinion (np.array): The majority opinion inserted in the test case.

        Returns:
            np.array: The metric value for the test case.
        """
        metric_value = np.sum(
            (control_answer == treatment_answer) & (control_answer == majority_opinion)
        ) - np.sum(
            (control_answer == treatment_answer) & (control_answer != majority_opinion)
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
            majority_opinion = np.array(
                [
                    [0 if test_case.CONTROL_VALUES["majority_opinion"][0] == "A" else 1]
                    for (test_case, _) in test_results
                ]
            )
            biasedness_scores = np.mean(
                self._compute(
                    control_answer, treatment_answer, majority_opinion
                )
            )
        except Exception as e:
            print(e)
            raise MetricCalculationError(f"Error filtering test results: {e}")
        return biasedness_scores
