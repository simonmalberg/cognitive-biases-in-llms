from base import TestGenerator, LLM, Metric, MetricCalculationError
from tests import TestCase, Template, DecisionResult
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

    def _custom_population(
        self, completed_template: Template, custom_values: dict, seed: int
    ) -> None:
        """
        Custom population method for the Confirmation Bias test case.

        Args:
            completed_template (Template): The assembled template for the test case.
            custom_values (dict): The custom values for the test case.
            seed (int): The seed for the random number generator.
        """
        # Loading the options
        kinds = custom_values["kind"]
        random.seed(seed)
        # Sampling one of ['positive', 'negative']
        kind = random.choice(kinds)
        opposite_kind = kinds[0] if kind == kinds[1] else kinds[1]
        # Inserting the sample into the template
        completed_template.insert_values(
            list(zip(["kind", "opposite_kind"], [kind, opposite_kind])), kind="manual"
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


class ConfirmationBiasMetric(Metric):
    """
    A class that describes the quantitative evaluation of the confirmation bias in a model.

    Metric:
    𝔅 = â₂ * (2 * I[â₁=a] - 1)
    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    a is the opposite kind in the test case.

    TODO: both -1 and 1 metric value can be seen as an instance of the confirmation bias - discuss

    """

    def _compute(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        opposite_kind: np.array,
    ) -> np.array:
        """
        Compute the metric for the Confirmation bias.

        Args:
            control_answer (np.array): The answer chosen in the control version.
            treatment_answer (np.array): The answer chosen in the treatment version.
            opposite_kind (np.array): The index of the opposite kind in the test case.

        Returns:
            np.array: The metric value for the test case.
        """
        metric_value = treatment_answer * (2 * (control_answer == opposite_kind) - 1)

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
            # always yields 0 for Yes and 1 for No
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
            # always yields 0 for Yes and 1 for No
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
            opposite_kind_idx = np.array(
                [
                    [
                        (
                            0
                            if test_case.TREATMENT_VALUES["opposite_kind"][0]
                            == "Positive"
                            else 1
                        )
                    ]
                    for (test_case, _) in test_results
                ]
            )
            biasedness_scores = np.mean(
                self._compute(
                    control_answer_idx, treatment_answer_idx, opposite_kind_idx
                )
            )
        except Exception as e:
            print(e)
            raise MetricCalculationError(f"Error filtering test results: {e}")
        return biasedness_scores
