from base import TestGenerator, LLM, Metric, PopulationError, MetricCalculationError
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np


class StatusQuoBiasTestGenerator(TestGenerator):
    """
    Test generator for the Status Quo Bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Status Quo Bias"
        self.config: TestConfig = super().load_config(self.BIAS)

    def generate_all(self, model: LLM, scenarios: list[str], seed: int = 42) -> list[TestCase]:
        # Load the custom values from the test configuration
        custom_values = self.config.get_custom_values()

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

        # Generate two decision alternatives the manager could choose from
        alternatives = self._create_alternatives(model, scenario, seed)

        # Insert for how many years the status quo has been in place
        number_of_years = config_values["number_of_years"]
        control.insert("number_of_years", str(number_of_years))
        treatment.insert("number_of_years", str(number_of_years))

        # Insert the alternatives into the templates
        control.insert("status A", alternatives[0])
        control.insert("status B", alternatives[1])
        treatment.insert("status A", alternatives[0])
        treatment.insert("status B", alternatives[1])

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

    def _create_alternatives(self, model: LLM, scenario: str, seed: int = 42) -> tuple[str, str]:
        """
        Creates two decision alternatives for the given scenario.

        Args:
            model (LLM): The language model to use for generating the alternatives.
            scenario (str): The scenario for which the alternatives are to be generated.
            seed (int, optional): The random seed to use for generating the alternatives. Defaults to 42.

        Returns:
            tuple[str, str]: A tuple containing the two generated alternatives.
        """
        
        # Define the prompt for generating the alternatives
        prompt = f"Scenario: {scenario}\n\nFor the given scenario, propose two alternative states for the organization, describing sustained, long-term initiatives rather than one-time activities. \
            Ensure both states provide equal value to the organization. Respond with two short lines of text, one for each state, separated by a line break. Do not enumerate the two states."

        # Generate the alternatives using the model
        response = model.prompt(prompt=prompt, seed=seed)

        # Split the response into the two alternatives and remove any empty alternatives
        alternatives = response.split("\n")
        alternatives = [a.strip() for a in alternatives if a.strip() != ""]

        # Check that the model generated exactly two alternatives
        if len(alternatives) != 2:
            raise PopulationError(f"Model was supposed to generate two alternatives, but generated {len(alternatives)} alternatives.", model_output=response)

        return alternatives[0], alternatives[1]


class StatusQuoBiasMetric(Metric):

    def __init__(self):
        pass

    def _compute(self, test_result: tuple[TestCase, DecisionResult]) -> float:
        # Extract the decision result from the tuple
        decision_result: DecisionResult = test_result[1]

        # Validate that only valid answer options were chosen
        if decision_result.CONTROL_DECISION not in [1, 2]:
            raise MetricCalculationError(f"Invalid answer option chosen: {decision_result.CONTROL_DECISION}. Only 1 and 2 are valid answer options.")
        if decision_result.TREATMENT_DECISION not in [1, 2]:
            raise MetricCalculationError(f"Invalid answer option chosen: {decision_result.TREATMENT_DECISION}. Only 1 and 2 are valid answer options.")

        # Extract the control and treatment answers from the decision result
        control_switch = decision_result.CONTROL_DECISION == 2       # True if control answer was to switch to the alternative
        treatment_switch = decision_result.TREATMENT_DECISION == 1   # True if treatment answer was to switch to the alternative

        # Calculate the biasedness
        if control_switch:
            if treatment_switch:
                biasedness = -1.0
            else:
                biasedness = 0.0
        else:
            if treatment_switch:
                biasedness = 0.0
            else:
                biasedness = 1.0

        return biasedness

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        # Calculate the average biasedness score across all tests
        biasedness_scores = [self._compute(test_result) for test_result in test_results]
        return np.mean(biasedness_scores)