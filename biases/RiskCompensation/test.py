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
        control: Template = self.config.get_control_template()      
        treatment: Template = self.config.get_treatment_template()  

        # Sample prior confidence in course of action
        min, max, step = config_values["initial_risk"]
        initial_risk = np.random.choice(np.arange(int(min), int(max)+1, int(step)), size=1)[0]

        # Sample posteriors lower and higher than prior
        min, max, step = config_values["risk_reduction"]
        risk_reduction = np.random.choice(np.arange(int(min), int(initial_risk)-10, int(step)), size=1)[0]

        # Insert the sampled values into the control template
        control.insert('initial_risk', str(initial_risk)+"%", origin='user')

        # Insert the sampled values into the treatment template
        treatment.insert('initial_risk', str(initial_risk)+"%", origin='user')
        treatment.insert('risk_reduction', str(risk_reduction)+"%", origin='user')

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
        
        pass

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        
        pass