from base import TestGenerator, LLM, Metric, MetricCalculationError
from tests import TestCase, Template, TestConfig, DecisionResult
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

    def _custom_population(
        self, completed_template: Template, custom_values: dict, seed: int
    ) -> None:
        """
        Custom population method for the Loss Aversion test case.

        Args:
            completed_template (Template): The assembled template for the test case.
            custom_values (dict): The custom values for the test case.
            seed (int): The seed for the random number generator.
        """
        # Loading the possible amounts and lambda values
        amount_values = custom_values['base_amount']
        lambda_values = custom_values['lambda_coef']
        np.random.seed(seed)
        # Loading the mean and max interval for the lambda coefficient
        lambda_min, lambda_max = float(lambda_values[1]), float(
            lambda_values[2]
        )
        # Loading the required distribution (should be a np.random method)
        lambda_distribution = getattr(np.random, lambda_values[0])
        # Sampling a numerical value
        lambda_coef = round(lambda_distribution(lambda_min, lambda_max, ), 1)
        # Sampling the base amount
        base_distribution = getattr(np.random, amount_values[0])
        base_amount = base_distribution(float(amount_values[1]), float(amount_values[2]))
        # Taking the respective amounts: base and lambda ones
        lambda_amount, base_amount = str(round(base_amount * lambda_coef, 1)), str(base_amount)

        # Inserting the values into the template
        patterns = ['lambda_amount', 'base_amount']
        values = [lambda_amount, base_amount]
        completed_template.insert_values(list(zip(patterns, values)), kind='manual')

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
        # Load the treatment template
        treatment: Template = self.config.get_treatment_template()
        # Populate the templates with custom values
        self._custom_population(treatment, custom_values, seed)
        # Get dictionary of inserted values
        treatment_values = treatment.inserted_values
        
        # Populate the template using the model and the scenario
        _, treatment = super().populate(model, None, treatment, scenario)

        # Create a test case object and remember the sampled lambda value
        test_case = TestCase(
            bias=self.BIAS,
            control=None,
            treatment=treatment,
            generator=model.NAME,
            scenario=scenario,
            control_values=None,
            treatment_values=treatment_values,
            variant=None,
            remarks=None,
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
    """

    def _compute(self, answer: np.array, lambda_val: np.array) -> np.array:
        """
        Computes the loss aversion bias metric for the given batch of test instances.

        Args:
            answer (np.array, shape (batch, 1)): The answer(s) chosen.
            lambda_val (np.array, shape (batch, 1)): The loss aversion hyperparameter(s).

        Returns:
            The loss aversion bias metric value.
        """
        result = 1 - np.sum(answer / lambda_val) / np.sum(1 / lambda_val)

        return result

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        try:
            # make sure all pairs are not None
            test_results = [
                pair for pair in test_results if pair[0] is not None and pair[1] is not None
            ]
            # extract lambda parameters from the test cases
            lambdas = (
                np.array(
                [
                    float(test_case.TREATMENT_VALUES['lambda_amount'][0]) / float(test_case.TREATMENT_VALUES['base_amount'][0])
                    for (test_case, _) in test_results
                ]
                )
            )
            # extract answers (-1 because the option indices are 1-indexed)
            treatment_answer = (
                np.array(
                    [
                        # 1 if selected risky option
                        1 if 'another' in decision_result.TREATMENT_OPTIONS[decision_result.TREATMENT_DECISION - 1] else 0
                        for (_, decision_result) in test_results
                    ]
                )
            )
            biasedness_scores = np.mean(
                self._compute(treatment_answer, lambdas)
            )
        except Exception as e:
            raise MetricCalculationError("The metric could not be computed.")
        return np.around(biasedness_scores, 2)