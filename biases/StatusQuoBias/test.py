from base import TestGenerator, LLM, RatioScaleMetric, PopulationError
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

    def generate(self, model: LLM, scenario: str, custom_values: dict = {}, temperature: float = 0.0, seed: int = 42) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # Generate two decision alternatives the manager could choose from
        alternatives = self._create_alternatives(model, scenario, seed)

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
            temperature=temperature,
            seed=seed,
            scenario=scenario,
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


class StatusQuoBiasMetric(RatioScaleMetric):
    """
    A metric that measures the presence and strength of Status-Quo Bias based on a set of test results.

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results, flip_treatment=True)