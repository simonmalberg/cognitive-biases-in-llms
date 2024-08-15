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

    def _custom_population(
        self, completed_template: Template, custom_values: dict, seed: int
    ) -> None:
        """
        Custom population method for the Bandwagon Effect test case.

        Args:
            completed_template (Template): The assembled template for the test case.
            custom_values (dict): The custom values for the test case.
            seed (int): The seed for the random number generator.
        """
        # Loading the options
        majority_options = custom_values["majority_option"]
        random.seed(seed)
        # Sampling one of ['A', 'B']
        majority_option = random.choice(majority_options)
        # Inserting the sample into the template
        completed_template.insert_values(
            list(zip(["majority_option"], [majority_option])), kind="manual"
        )

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
        self._custom_population(control, custom_values, seed)
        self._custom_population(treatment, custom_values, seed)
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
    a is the majority option inserted in the test case.

    """

    def _compute(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        majority_option: np.array,
    ) -> np.array:
        """
        Compute the metric for the Bandwagon effect.

        Args:
            control_answer (np.array): The answer chosen in the control version.
            treatment_answer (np.array): The answer chosen in the treatment version.
            majority_option (np.array): The majority option inserted in the test case.

        Returns:
            np.array: The metric value for the test case.
        """
        metric_value = np.sum(
            (control_answer == treatment_answer) & (control_answer == majority_option)
        ) - np.sum(
            (control_answer == treatment_answer) & (control_answer != majority_option)
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
            # extract indices of the chosen answers (-1 because the option indices are 1-indexed)
            control_answer_idx = np.array(
                [
                    [
                        decision_result.CONTROL_OPTION_ORDER.index(
                            decision_result.CONTROL_DECISION - 1
                        )
                    ]
                    for (_, decision_result) in test_results
                ]
            )
            treatment_answer_idx = np.array(
                [
                    [
                        decision_result.TREATMENT_OPTION_ORDER.index(
                            decision_result.TREATMENT_DECISION - 1
                        )
                    ]
                    for (_, decision_result) in test_results
                ]
            )
            majority_option_idx = np.array(
                [
                    [0 if test_case.CONTROL_VALUES["majority_option"][0] == "A" else 1]
                    for (test_case, _) in test_results
                ]
            )
            biasedness_scores = np.mean(
                self._compute(
                    control_answer_idx, treatment_answer_idx, majority_option_idx
                )
            )
        except Exception as e:
            print(e)
            raise MetricCalculationError(f"Error filtering test results: {e}")
        return biasedness_scores
