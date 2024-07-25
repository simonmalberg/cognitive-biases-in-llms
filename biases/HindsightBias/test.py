from base import TestGenerator, LLM, Metric, MetricCalculationError
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np
import re


class HindsightBiasTestGenerator(TestGenerator):
    """
    Test generator for the Hindsight Bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Hindsight Bias"
        self.config: TestConfig = super().load_config(self.BIAS)

    def _custom_population(
        self, completed_template: Template, custom_values: dict, seed: int
    ) -> None:
        """
        Custom population method for the Hindsight Bias test case.

        Args:
            completed_template (Template): The assembled template for the test case.
            custom_values (dict): The custom values for the test case.
            seed (int): The seed for the random number generator.
        """
        # Loading the mean and max interval for the samples of numerical value
        sample_min, sample_max = int(custom_values["percentage"][1]), int(
            custom_values["percentage"][2]
        )
        # Loading the required distribution (should be a np.random method)
        distribution = getattr(np.random, custom_values["percentage"][0])

        # Sampling a numerical value
        sample = str(int(distribution(sample_min, sample_max)))

        # Inserting the sample into the template
        completed_template.insert_values(
            list(zip(["percentage"], [sample])), kind="manual"
        )

        return sample

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
        _ = self._custom_population(treatment, custom_values, seed)
        # Get dictionary of inserted values
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
            control_values=None,
            treatment_values=treatment_values,
            variant=None,
            remarks=None,
        )

        return test_case


class HindsightBiasMetric(Metric):
    """
    A class that describes the quantitative evaluation of the optimism bias in a model.

    Individual metric:
    𝔅 = sgn(x) * min(‖x‖₁, 1), x = (â₂ − â₁) / (a − â₁)

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    a is the ground truth percentage (sampled using custom values).

    """

    def _compute(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        ground_truth: np.array,
    ) -> np.array:
        """
        Compute the metric for the hindsight bias.

        Args:
            control_answer (np.array): The answer chosen in the control version.
            treatment_answer (np.array): The answer chosen in the treatment version.
            answer_options (np.array): The answer options for the test case.

        Returns:
            np.array: The metric value for the test case.
        """
        # Calculate the metric value
        metric_value = np.clip(
            (treatment_answer - control_answer)
            / (ground_truth - control_answer + 1e-8),
            -1,
            1,
        )

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
                int(re.findall(r"-?\d+\.?\d*", s)[0]) for s in options.values()
            ]
            if not answer_options.size:
                answer_options = np.array([numerical_options])
            else:
                answer_options = np.vstack((answer_options, numerical_options))

        return answer_options

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        # make sure all pairs are not None
        test_results = [
            pair for pair in test_results if pair[0] is not None and pair[1] is not None
        ]
        try:
            # extract answer options from the test results
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
            # extract the ground truth values
            ground_truth = np.array(
                [
                    int(test_case.TREATMENT_VALUES["percentage"][0])
                    for (test_case, _) in test_results
                ]
            )
            biasedness_scores = np.mean(
                self._compute(control_answer, treatment_answer, ground_truth)
            )
        except Exception as e:
            raise MetricCalculationError("The metric could not be computed.")
        return np.around(biasedness_scores, 2)
