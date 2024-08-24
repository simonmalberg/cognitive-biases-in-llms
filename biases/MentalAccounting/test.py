from base import TestGenerator, LLM, Metric, MetricCalculationError
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np


class MentalAccountingTestGenerator(TestGenerator):
    """
    Test generator for the Mental Accounting.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Mental Accounting"
        self.config: TestConfig = super().load_config(self.BIAS)

    def generate_all(
        self, model: LLM, scenarios: list[str], seed: int = 42
    ) -> list[TestCase]:
        # Load the custom values from the test config
        custom_values = (
            self.config.get_custom_values()
        )  # TODO: Remove this line if custom values are not needed

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
        np.random.seed(seed)
        price_values = custom_values["price_thousands"]
        # Loading the mean and max interval for the price
        price_min, price_max = float(price_values[1]), float(price_values[2])
        # Loading the required distribution (should be a np.random method)
        price_distribution = getattr(np.random, price_values[0])
        # Sampling a numerical value for the price
        price = str(price_distribution(price_min, price_max) * 1000)
        # Inserting the sampled value into the templates
        control.insert("price", price, origin="user")
        treatment.insert("price", price, origin="user")

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
            treatment_values=None,
            variant=None,
            remarks=None,
        )

        return test_case


class MentalAccountingMetric(Metric):
    """
    A class that describes the quantitative evaluation of the Mental Accounting in a model.

    Metric:
    𝔅 = â₁ - â₂ ∈ {-1, 0, 1}
    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    """

    def _compute(
        self, control_answer: np.array, treatment_answer: np.array
    ) -> np.array:
        """
        Compute the metric for the Mental Accounting.

        Args:
            control_answer (np.array): The answer chosen in the control version.
            treatment_answer (np.array): The answer chosen in the treatment version.

        Returns:
            np.array: The metric value for the test case.
        """
        metric_value = control_answer - treatment_answer

        return metric_value

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        try:
            # make sure all pairs are not None
            test_results = [
                pair
                for pair in test_results
                if pair[0] is not None and pair[1] is not None
            ]
            # extract chosen answers
            control_answer = np.array(
                [
                    decision_result.CONTROL_DECISION
                    for (_, decision_result) in test_results
                ]
            )
            treatment_answer = np.array(
                [
                    decision_result.TREATMENT_DECISION
                    for (_, decision_result) in test_results
                ]
            )
            # compute the biasedness scores
            biasedness_scores = np.mean(self._compute(control_answer, treatment_answer))
        except Exception as e:
            print(e)
            raise MetricCalculationError(f"Error filtering test results: {e}")
        return biasedness_scores
