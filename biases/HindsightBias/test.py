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

        # Populate the treatment template with a custom value
        np.random.seed(seed)
        # Loading the mean and max interval for the sample of numerical value
        sample_min, sample_max = int(custom_values["percentage"][1]), int(
            custom_values["percentage"][2]
        )
        # Loading the required distribution (should be a np.random method)
        distribution = getattr(np.random, custom_values["percentage"][0])
        # Sampling a numerical value
        sample = str(
            int(
                distribution(
                    sample_min,
                    sample_max,
                )
            )
        )
        # Inserting the sample into the treatment template
        treatment.insert("percentage", sample, origin="user")

        # Populate the templates using the model and the scenario
        control, treatment = super().populate(model, control, treatment, scenario)

        # Create a test case object
        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=treatment,
            generator=model.NAME,
            scenario=scenario,
            variant=None,
            remarks=None,
        )

        return test_case


class HindsightBiasMetric(Metric):
    """
    A class that describes the quantitative evaluation of the hindsight bias in a model.

    Metric:
    𝔅 = (‖ â₁ − a' ‖₁ − ‖ â₂ − a' ‖₁) / a ∈ [-1, 1];

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    a' is the option closest to the ground truth percentage (sampled using custom values);
    a = max[‖ â₁ − a' ‖₁, ‖ â₂ − a' ‖₁].
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
            ground_truth (np.array): The option closest to the ground truth percentage value for the test case.

        Returns:
            np.array: The metric value for the test case.
        """
        # Calculate the metric value
        delta_control = np.abs(control_answer - ground_truth)
        delta_treatment = np.abs(treatment_answer - ground_truth)
        metric_value = (delta_control - delta_treatment) / np.maximum(
            delta_control, delta_treatment
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
            numerical_options = [int(re.findall(r"-?\d+\.?\d*", s)[0]) for s in options]
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
            # extract the chosen answers
            control_answer = np.take_along_axis(answer_options, control_answer, axis=1)
            treatment_answer = np.take_along_axis(
                answer_options, treatment_answer, axis=1
            )
            # extract the ground truth values
            ground_truth = [
                [
                    insertion.text
                    for insertion in test_case.TREATMENT.get_insertions()
                    if insertion.pattern == "percentage"
                ]
                for (test_case, _) in test_results
            ]
            ground_truth = np.array(
                [[round(int(p[0]) / 10) * 10] for p in ground_truth]
            )
            biasedness_scores = np.mean(
                self._compute(control_answer, treatment_answer, ground_truth)
            )
        except Exception as e:
            print(e)
            raise MetricCalculationError(f"Error computing the metric: {e}")
        return round(biasedness_scores, 2)
