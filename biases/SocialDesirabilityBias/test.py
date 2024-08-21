from base import TestGenerator, LLM, Metric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np
import random


class SocialDesirabilityBiasTestGenerator(TestGenerator):
    """
    Test generator for the Social Desirability Bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Social Desirability Bias"
        self.config: TestConfig = super().load_config(self.BIAS)

    def generate_all(self, model: LLM, scenarios: list[str], seed: int = 42) -> list[TestCase]:
        # Load the custom values from the test config
        config_values = self.config.get_custom_values()

        # Create test cases for all scenarios
        test_cases: list[TestCase] = []
        for scenario in scenarios:
            try:
                statement, desirable = self._get_random_statement(config_values["statements"], scenario + str(seed))
                custom_values = {
                    "statement": statement,
                    "desirable": desirable
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

        # Extract the config values
        statement = config_values["statement"]
        desirable = config_values["desirable"]

        # Insert the statement into the templates
        control.insert("statement", statement, origin='user')
        treatment.insert("statement", statement, origin='user')

        # Insert a verb signalling that the rational behavior is the opposite of the socially desirable behavior
        treatment.insert("desirable", "comply" if desirable else "not comply", origin='user')
        treatment.insert("undesirable", "not comply" if desirable else "comply", origin='user')

        # Insert the answer options into the treatment template
        control.insert("desirable_answer", "True" if desirable else "False", origin='user')
        control.insert("undesirable_answer", "False" if desirable else "True", origin='user')

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
            remarks=f"Statement used: '{statement}'. Socially desirable answer: '{desirable}'"
        )

        return test_case

    def _get_random_statement(self, from_statements: list[str], seed: int) -> tuple[str, bool]:
        # Sample a random statement from the list of statements
        random.seed(seed)
        statement = random.choice(from_statements)

        # Split off the last three characters from the statement string which contain the label ('(T)' or '(F)') signalling whether the statement is socially desirable or undesirable
        desirable = True if statement[-3:] == '(T)' else False
        statement = statement[:-4]
        
        return statement, desirable


class SocialDesirabilityBiasMetric(Metric):

    def __init__(self):
        pass

    def _compute(self, test_result: tuple[TestCase, DecisionResult]) -> float:
        # Extract the decision result from the tuple
        decision_result: DecisionResult = test_result[1]

        # Parse whether the control and treatment decisions correspond to the socially desirable option
        DESIRABLE_ANSWER_CTRL = 1
        DESIRABLE_ANSWER_TRMT = 1
        control_desirable = True if decision_result.CONTROL_DECISION == DESIRABLE_ANSWER_CTRL else False
        treatment_desirable = True if decision_result.TREATMENT_DECISION == DESIRABLE_ANSWER_TRMT else False

        # Calculate the biasedness
        if control_desirable:
            if treatment_desirable:
                biasedness = 0.0
            else:
                biasedness = 1.0
        else:
            if treatment_desirable:
                biasedness = -1.0
            else:
                biasedness = 0.0

        return biasedness

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        # Calculate the average biasedness score across all tests
        biasedness_scores = [self._compute(test_result) for test_result in test_results]
        return np.mean(biasedness_scores)