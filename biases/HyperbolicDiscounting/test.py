from base import TestGenerator, LLM, Metric, MetricCalculationError
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np


class HyperbolicDiscountingTestGenerator(TestGenerator):
    """
    Test generator for the Hyperbolic Discounting.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Hyperbolic Discounting"
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
        np.random.seed(seed)
        earlier_values, later_values, delay_values, scheme_control, scheme_treatment = (
            custom_values["earlier_reward"],
            custom_values["later_coef"],
            custom_values["months_delay"],
            custom_values['scheme_control'],
            custom_values['scheme_treatment']
        )
        # Sampling the order of the schemes
        index = np.random.choice(range(len(scheme_control)))
        for template in [control, treatment]:
            template.insert("control_scheme", scheme_control[index], origin="user")
            template.insert("other_control_scheme", scheme_control[1 - index], origin="user")
            template.insert("treatment_scheme", scheme_treatment[index], origin="user")
            template.insert("other_treatment_scheme", scheme_treatment[1 - index], origin="user")
            
        # Loading the required distributions (should be np.random methods)
        earlier_distribution, later_distribution, delay_distribution = (
            getattr(np.random, earlier_values[0]),
            getattr(np.random, later_values[0]),
            getattr(np.random, delay_values[0]),
        )
        # Sampling the respective numerical values
        earlier_reward = earlier_distribution(
            int(earlier_values[1]), int(earlier_values[2])
        )
        later_reward = min(
            100,
            int(
                earlier_reward
                * later_distribution(int(later_values[1]), int(later_values[2]))
            ),
        )
        months_delay = delay_distribution(int(delay_values[1]), int(delay_values[2]))
        # Inserting the sampled value into the template
        for template in [control, treatment]:
            template.insert("earlier_reward", str(earlier_reward), origin="user")
            template.insert("later_reward", str(later_reward), origin="user")
            template.insert("months_delay", str(months_delay), origin="user")

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


class HyperbolicDiscountingMetric(Metric):
    """
    A class that describes the quantitative evaluation of the Hyperbolic Discounting in a model.

    Metric:
    𝔅 = â₂ - â₁ ∈ {-1, 0, 1}
    where:
    â₂, â₁ are the chosen answers for the treatment and control versions, respectively.
    """

    def _compute(
        self, control_answer: np.array, treatment_answer: np.array
    ) -> np.array:
        """
        Compute the metric for the Hyperbolic Discounting.

        Args:
            control_answer (np.array): The answer chosen in the control version.
            treatment_answer (np.array): The answer chosen in the treatment version.

        Returns:
            np.array: The metric value for the test case.
        """
        metric_value = treatment_answer - control_answer

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
            raise MetricCalculationError(f"Error computing the metric: {e}")
        return biasedness_scores
