from base import TestGenerator, LLM, Metric, MetricCalculationError
from tests import TestCase, Template, TestConfig, DecisionResult
import random
import numpy as np
import re


class AvailabilityHeuristicTestGenerator(TestGenerator):
    """
    Test generator for the Availability heuristic.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Availability Heuristic"
        self.config: TestConfig = super().load_config(self.BIAS)

    def _custom_population(
        self, completed_template: Template, custom_values: dict, seed: int
    ) -> None:
        """
        Custom population method for the Availability Heuristic test case.

        Args:
            completed_template (Template): The assembled template for the test case.
            custom_values (dict): The custom values for the test case.
            seed (int): The seed for the random number generator.
        """
        # Loading the possible outcomes
        outcomes = custom_values["outcome"]
        random.seed(seed)
        # Sampling one of ['failure', 'success']
        chosen_outcome = random.choice(outcomes)
        # Inserting the sample into the template
        completed_template.insert_values(
            list(zip(["outcome"], [chosen_outcome])), kind="manual"
        )

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


class AvailabilityHeuristicMetric(Metric):
    """
    A class that describes the quantitative evaluation of the availability heuristic in a model.

    Metric:
    𝔅 = (â₂ - â₁) / a

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    a = â₁ - â (if â₂ - â₁ < 0) or else a = ã - â₁, where ã is the maximum option, â - the minimum option.

    """

    def _compute(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        max_option: np.array, 
        min_option: np.array
    ) -> np.array:
        """
        Compute the metric for the Availability Heuristic.

        Args:
            control_answer (np.array): The answer chosen in the control version.
            treatment_answer (np.array): The answer chosen in the treatment version.
            max_option (np.array): The maximum answer option.
            min_option (np.array): The minimum answer option.

        Returns:
            np.array: The metric value for the test case.
        """
        delta = treatment_answer - control_answer
        metric_value = delta / ((delta >= 0) * (max_option - control_answer) + (delta < 0) * (control_answer - min_option) + 10e-8)

        return metric_value

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
                int(re.findall(r"-?\d+\.?\d*", s)[0]) for s in options
            ]
            if not answer_options.size:
                answer_options = np.array([numerical_options])
            else:
                answer_options = np.vstack((answer_options, numerical_options))

        return answer_options

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        try:
            # make sure all pairs are not None
            test_results = [
                pair for pair in test_results if pair[0] is not None and pair[1] is not None
            ]
            # extract answer options from the test results
            answer_options = self.assemble_options(
                [
                    decision_result.CONTROL_OPTIONS
                    for (_, decision_result) in test_results
                ]
            )
            max_option = np.max(answer_options, axis=1)
            min_option = np.min(answer_options, axis=1)
            # extract indices of the chosen answers (-1 because the option indices are 1-indexed)
            control_answer_idx = (
                np.array(
                    [
                        [decision_result.CONTROL_DECISION]
                        for (_, decision_result) in test_results
                    ]
                )
                - 1
            )
            treatment_answer_idx = (
                np.array(
                    [
                        [decision_result.TREATMENT_DECISION]
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
            biasedness_scores = np.mean(
                self._compute(control_answer, treatment_answer, max_option, min_option)
            )
        except Exception as e:
            print(e)
            raise MetricCalculationError("The metric could not be computed.")
        return round(biasedness_scores, 2)
