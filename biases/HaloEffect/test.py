from base import TestGenerator, LLM, Metric, MetricCalculationError
from tests import TestCase, Template, TestConfig, DecisionResult
import random
import numpy as np


class HaloEffectTestGenerator(TestGenerator):
    """
    Test generator for the Halo Effect bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Halo Effect bias.
    """

    def __init__(self):
        self.BIAS = "Halo Effect"
        self.config = super().load_config(self.BIAS)

    def _custom_population(
        self, completed_template: Template, custom_values: dict, seed: int
    ) -> None:
        """
        Custom population method for the Halo Effect test case.

        Args:
            completed_template (Template): The assembled template for the test case.
            custom_values (dict): The custom values for the test case.
            seed (int): The seed for the random number generator.
        """
        # Loading the dict with custom values
        random.seed(seed)
        perception_values = custom_values["perception"]
        preference_values = custom_values["preference"]
        # Sampling one of the values
        perception = random.choice(perception_values)
        preferential_brand = random.choice(preference_values)
        other_brand = (
            preference_values[0]
            if preferential_brand == preference_values[1]
            else preference_values[1]
        )
        perceived_brand = (
            other_brand if perception == "positive" else preferential_brand
        )
        completed_template.insert_values(
            list(
                zip(
                    ["perception", "preference", "perceived_brand"],
                    [perception, preferential_brand, perceived_brand],
                )
            ),
            kind="manual",
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
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # sample and insert a sentiment for treatment version
        self._custom_population(control, custom_values, seed)
        self._custom_population(treatment, custom_values, seed)
        # get dictionary of inserted values
        control_values = control.inserted_values
        treatment_values = treatment.inserted_values

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


class HaloEffectMetric(Metric):
    """
    A class that describes the quantitative evaluation of the halo effect bias in a model.

    Metric:
    𝔅 = I[â₂ != â₁]

    where:
    I is the indicator function;
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively.

    """

    def _compute(
        self, control_answer: np.array, treatment_answer: np.array
    ) -> np.array:
        """
        Computes the halo effect bias metric for the given batch of test instances.

        Args:
            control_answer (np.array, shape (batch, 1)): The answer(s) chosen for the control variant.
            treatment_answer (np.array, shape (batch, 1)): The answer(s) chosen for the treatment variant.

        Returns:
            The halo effect bias metric value.
        """
        metric_values = 1 * (control_answer != treatment_answer)

        return metric_values

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        try:
            # make sure all pairs are not None
            test_results = [
                pair
                for pair in test_results
                if pair[0] is not None and pair[1] is not None
            ]
            # extract answers
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
            biasedness_scores = np.mean(self._compute(control_answer, treatment_answer))
        except Exception as e:
            print(e)
            raise MetricCalculationError(f"Error filtering test results: {e}")
        return biasedness_scores
