from base import TestGenerator, LLM, Metric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np
import random


class StereotypingTestGenerator(TestGenerator):
    """
    Test generator for the Stereotyping.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Stereotyping"
        self.config: TestConfig = super().load_config(self.BIAS)

    def generate_all(self, model: LLM, scenarios: list[str], seed: int = 42) -> list[TestCase]:
        # Load the custom values from the test config
        config_values = self.config.get_custom_values()   # TODO: Remove this line if custom values are not needed

        # Create test cases for all scenarios
        test_cases: list[TestCase] = []
        for scenario in scenarios:
            try:
                random.seed(scenario + str(seed))
                group = random.choice(config_values["group"])
                custom_values = {
                    "group": group
                }

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

        # Populate the templates with custom values
        control.insert("group", config_values["group"], origin='user')
        treatment.insert("group", config_values["group"], origin='user')

        # Populate the templates using the model and the scenario
        treatment, control = super().populate(model, treatment, control, scenario)

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


class StereotypingMetric(Metric):

    def __init__(self):
        pass

    def _compute(self, test_result: tuple[TestCase, DecisionResult]) -> float:
        # Extract the test case and decision result from the tuple
        test_case: TestCase = test_result[0]
        decision_result: DecisionResult = test_result[1]
        
        # Determine if the chosen control and treatment options correspond to a characteristic that is stereotypical of the group
        control_options, _ = test_case.CONTROL.get_options(shuffle_options=False, apply_insertions=False)
        control_options = [(True if "stereotypical" in option else False) for option in control_options]
        control_stereotypical = control_options[decision_result.CONTROL_DECISION]

        treatment_options, _ = test_case.TREATMENT.get_options(shuffle_options=False, apply_insertions=False)
        treatment_options = [(True if "stereotypical" in option else False) for option in treatment_options]
        treatment_stereotypical = treatment_options[decision_result.TREATMENT_DECISION]

        # Determine the biasedness
        if control_stereotypical:
            if treatment_stereotypical:
                biasedness = 0.0
            else:
                biasedness = -1.0
        else:
            if treatment_stereotypical:
                biasedness = 1.0
            else:
                biasedness = 0.0

        return biasedness

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        # Calculate the average biasedness score across all tests
        biasedness_scores = [self._compute(test_result) for test_result in test_results]
        return np.mean(biasedness_scores)