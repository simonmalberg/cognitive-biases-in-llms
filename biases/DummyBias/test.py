from base import TestGenerator, LLM
from tests import TestCase, Template, TestConfig


class DummyBiasTestGenerator(TestGenerator):
    """
    Dummy test generator for generating test cases. This class is implemented for testing purposes only.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Dummy Bias.
    """

    def __init__(self):
        self.BIAS = "Dummy Bias"
        self.config = super().load_config(self.BIAS)

    def generate_all(self, model: LLM, scenarios: list[str], config_values: dict, seed: int) -> list[TestCase]:
        # TODO Implement functionality to generate multiple test cases at once (potentially following the ranges or distributions outlined in the config values)
        pass

    def generate(self, model: LLM, scenario: str, config_values: dict, seed: int) -> TestCase:
        # TODO Refactor to use only the config values passed to this method (i.e., only the values to be applied to the generation of this very test case)
        
        # Load the control and treatment templates from the test configuration
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # Populate the templates using the model and scenario
        control, treatment = super().populate(model, control, treatment, scenario)

        # Create a test case object
        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=treatment,
            generator=model.NAME,
            control_custom_values=None,
            treatment_custom_values=None,
            scenario=scenario
        )

        return test_case