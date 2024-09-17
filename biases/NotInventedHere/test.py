from base import TestGenerator, LLM, Metric, PopulationError, MetricCalculationError
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np


class NotInventedHereTestGenerator(TestGenerator):
    """
    Test generator for the Not Invented Here Syndrome.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Not Invented Here"
        self.config: TestConfig = super().load_config(self.BIAS)

    def generate_all(self, model: LLM, scenarios: list[str], seed: int = 42) -> list[TestCase]:

        # Get a list of all variants in the test config
        variants = self.config.get_variants()

        # Create test cases for all variants and scenarios
        test_cases: list[TestCase] = []
        for variant in variants:
            for scenario in scenarios:
                try:
                    instance_values = {
                        "variant": variant
                    }

                    test_case = self.generate(model, scenario, instance_values, seed)
                    test_cases.append(test_case)
                except Exception as e:
                    print("Generating the test case failed.")
                    print(f"Variant: {variant}")
                    print(f"Scenario: {scenario}")
                    print(f"Seed: {seed}")
                    print(e)

        return test_cases

    def generate(self, model: LLM, scenario: str, config_values: dict = {}, seed: int = 42) -> TestCase:


        # Load the control and treatment templates for the selected variant
        variant = config_values["variant"]
        control: Template = self.config.get_control_template(variant)
        treatment: Template = self.config.get_treatment_template(variant)

        # Generate two decision alternatives the manager could choose from
        alternatives = self._create_alternatives(model, scenario, seed)

        # Insert the alternatives into the templates
        control.insert("proposal A", alternatives[0])
        control.insert("proposal B", alternatives[1])
        treatment.insert("proposal A", alternatives[0])
        treatment.insert("proposal B", alternatives[1])

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
            variant=variant,
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
        prompt = f"Scenario: {scenario}\n\nFor the given scenario, describe two alternative proposals for the manager to consider. \
            Ensure both proposals provide equal value to the organization. Respond with two short lines of text, one for each proposal, separated by a line break. Do not enumerate the two states."

        # Generate the alternatives using the model
        response = model.prompt(prompt=prompt, seed=seed)

        # Split the response into the two alternatives and remove any empty alternatives
        alternatives = response.split("\n")
        alternatives = [a.strip() for a in alternatives if a.strip() != ""]

        # Check that the model generated exactly two alternatives
        if len(alternatives) != 2:
            raise PopulationError(f"Model was supposed to generate two alternatives, but generated {len(alternatives)} alternatives.", model_output=response)

        return alternatives[0], alternatives[1]


class NotInventedHereMetric(Metric):

    def __init__(self):
        pass

    def _compute(self, test_result: tuple[TestCase, DecisionResult]) -> float:
        # Extract the decision result from the tuple
        decision_result: DecisionResult = test_result[1]

        print("Metric not implemented yet.")
        pass

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        # Calculate the average biasedness score across all tests
        biasedness_scores = [self._compute(test_result) for test_result in test_results]
        return np.mean(biasedness_scores)