from base import TestGenerator, LLM, Metric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np
import random


class DispositionEffectTestGenerator(TestGenerator):
    """
    Test generator for the Disposition Effect.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Disposition Effect"
        self.config: TestConfig = super().load_config(self.BIAS)

    def generate_all(self, model: LLM, scenarios: list[str], seed: int = 42) -> list[TestCase]:
        # Load the custom values from the test config
        config_values = self.config.get_custom_values()

        # Create test cases for all scenarios
        test_cases: list[TestCase] = []
        for scenario in scenarios:
            try:
                test_case = self.generate(model, scenario, config_values, seed)
                test_cases.append(test_case)
            except Exception as e:
                print(f"Generating the test case failed.\nScenario: {scenario}\nSeed: {seed}")
                print(e)

        return test_cases

    def generate(self, model: LLM, scenario: str, config_values: dict = {}, seed: int = 42) -> TestCase:
        # Load the control and treatment templates
        treatment: Template = self.config.get_treatment_template()

        # Extract change_range from config_values
        change_range = config_values.get("change_range")
        min_change = int(change_range[0])
        max_change = int(change_range[1])
        
        # Initialize the random number generator with the combined seed
        rng = random.Random(hash(scenario) + seed)
        
        # Generate the increase and decrease values within the specified range
        increase = rng.randint(min_change, max_change)
        decrease = rng.randint(min_change, max_change)

        # Populate the templates with custom values
        treatment.insert("increase", str(increase), origin="user")
        treatment.insert("decrease", str(decrease), origin="user")

        # Populate the templates using the model and the scenario
        _, treatment = super().populate(model, None, treatment, scenario)

        # Create a test case object
        test_case = TestCase(
            bias=self.BIAS,
            control=None,
            treatment=treatment,
            generator=model.NAME,
            scenario=scenario,
            variant=None,
            remarks={"increase": increase, "decrease": decrease}
        )

        return test_case


class DispositionEffectMetric(Metric):

    def __init__(self):
        pass

    def _compute(self, test_result: tuple[TestCase, DecisionResult]) -> float:
        # Extract the decision
        decision_result: DecisionResult = test_result[1]
        decision = decision_result.TREATMENT_DECISION

        # Calculate the biasedness
        if decision == 0:
            biasedness = -1.0 # Continue holding high-performing asset A and sell low-performing asset B
        elif decision == 1:
            biasedness = 0.0 # Continue holding both assets
        elif decision == 2:            
            biasedness = 0.0 # Sell both assets
        elif decision == 3:            
            biasedness = 1.0 # Sell high-performing asset A and continue holding low-performing asset B

        return biasedness

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        # Calculate the average biasedness score across all tests
        biasedness_scores = [self._compute(test_result) for test_result in test_results]
        return np.mean(biasedness_scores)