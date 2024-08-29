from base import TestGenerator, LLM, Metric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np


class RiskCompensationTestGenerator(TestGenerator):
    """
    Test generator for the Risk Compensation.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Risk Compensation"
        self.config: TestConfig = super().load_config(self.BIAS)

    def generate_all(self, model: LLM, scenarios: list[str], seed: int = 42) -> list[TestCase]:
        # Load the custom values from the test config
        custom_values = self.config.get_custom_values()   # TODO: Remove this line if custom values are not needed

        # Create test cases for all scenarios
        test_cases: list[TestCase] = []
        for scenario in scenarios:
            try:
                test_case = self.generate(model, scenario, custom_values, seed)
                test_cases.append(test_case)
            except Exception as e:
                print(f"Generating the test case failed.\nScenario: {scenario}\nSeed: {seed}")
                print(e)

        return test_cases

    def generate(self, model: LLM, scenario: str, config_values: dict = {}, seed: int = 42) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()       # TODO: Pass the variant name as a function parameter if you have more than one test variant
        treatment: Template = self.config.get_treatment_template()   # TODO: Pass the variant name as a function parameter if you have more than one test variant

        # Populate the templates with custom values
        #treatment.insert([("risk_decrease", config_values["risk_decrease"])], kind='user')   # TODO: Remove this line if custom values are not needed
        #treatment.insert([("risk_increase", config_values["risk_increase"])], kind='user')   # TODO: Remove this line if custom values are not needed


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
            remarks=None
        )

        return test_case


class RiskCompensationMetric(Metric):

    def __init__(self):
        pass

    def _compute(self, test_result: tuple[TestCase, DecisionResult]) -> float:
        # Extract the decision result from the tuple
        decision_result: DecisionResult = test_result[1]

        # Extract the chosen option
        control_decision = int(decision_result.CONTROL_DECISION)
        treatment_decision = int(decision_result.TREATMENT_DECISION)

        biasedness_score = treatment_decision - control_decision

        # Calculate the biasedness
        return biasedness_score

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        biasedness_scores = [self._compute(test_result) for test_result in test_results]
        return np.around(biasedness_scores, 2)