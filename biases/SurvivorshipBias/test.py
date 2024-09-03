from base import TestGenerator, LLM, Metric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np
import random


class SurvivorshipBiasTestGenerator(TestGenerator):
    """
    Test generator for the Survivorship Bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Survivorship Bias"
        self.config: TestConfig = super().load_config(self.BIAS)

    def generate_all(self, model: LLM, scenarios: list[str], seed: int = 42) -> list[TestCase]:
        # Create test cases for all scenarios
        test_cases: list[TestCase] = []
        for scenario in scenarios:
            try:
                test_case = self.generate(model, scenario, {}, seed)
                test_cases.append(test_case)
            except Exception as e:
                print(f"Generating the test case failed.\nScenario: {scenario}\nSeed: {seed}")
                print(e)

        return test_cases

    def generate(self, model: LLM, scenario: str, config_values: dict = {}, seed: int = 42) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # Randomly sample two percentages signaling how common a certain characteristic is in the survivor and non-survivor group, respectively
        range_min, range_max = self.config.get_custom_values()["percentage_range"]
        range_min, range_max = int(range_min), int(range_max)
        random.seed(scenario + str(seed))
        survivor_percentage = random.randint(range_min, range_max)
        non_survivor_percentage = random.randint(range_min, range_max)

        # Insert the percentages into the templates
        control.insert("survivor_percentage", str(survivor_percentage), "user")
        control.insert("non_survivor_percentage", str(non_survivor_percentage), "user")
        treatment.insert("survivor_percentage", str(survivor_percentage), "user")

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


class SurvivorshipBiasMetric(Metric):

    def __init__(self):
        pass

    def _compute(self, test_result: tuple[TestCase, DecisionResult]) -> float:
        # Extract the decision result from the tuple
        decision_result: DecisionResult = test_result[1]

        # Extract the control and treatment decisions
        control_decision = decision_result.CONTROL_DECISION
        treatment_decision = decision_result.TREATMENT_DECISION

        # Get the number of available options in control and treatment
        SCALE_OPTIONS_CONTROL = len(decision_result.CONTROL_OPTIONS)
        SCALE_OPTIONS_TREATMENT = len(decision_result.TREATMENT_OPTIONS)

        # Calculate biasedness as the deviation between the control and treatment decision normalized by the maximum possible deviation
        if treatment_decision > control_decision:
            if SCALE_OPTIONS_TREATMENT - 1 - control_decision == 0:
                # Catch division by zero errors
                biasedness = 0.0
            else:
                biasedness = (treatment_decision - control_decision) / (SCALE_OPTIONS_TREATMENT - 1 - control_decision)
        else:
            if control_decision == 0:
                # Catch division by zero errors
                biasedness = 0.0
            else:
                biasedness = (treatment_decision - control_decision) / control_decision

        return biasedness

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        # Calculate the average biasedness score across all tests
        biasedness_scores = [self._compute(test_result) for test_result in test_results]
        return np.mean(biasedness_scores)