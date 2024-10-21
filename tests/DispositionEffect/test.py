from core.base import TestGenerator, LLM, RatioScaleMetric
from core.testing import TestCase, Template, TestConfig, DecisionResult
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

    def sample_custom_values(self, num_instances: int, iteration_seed: int) -> dict:
        """
        Sample custom values for the test case generation.

        Args:
            num_instances (int): The number of instances expected to be generated for each scenario.
            iteration_seed (int): The seed to use for sampling the custom values.

        Returns:
            dict: A dictionary containing the sampled custom values.
        """

        # Get the value range from the custom values configuration
        custom_values = self.config.get_custom_values()
        range_min, range_max = self.config.get_custom_values()["change_range"]
        range_min, range_max = int(range_min), int(range_max)

        # Randomly sample pairs of percentage values measuring how much the assets have increased or decreased, respectively
        random.seed(iteration_seed)
        increase = [random.randint(range_min, range_max) for _ in range(num_instances)]
        decrease = [random.randint(range_min, range_max) for _ in range(num_instances)]

        # Return a dictionary with the sampled values
        return {
            "increase": increase,
            "decrease": decrease
        }

    def generate(self, model: LLM, scenario: str, custom_values: dict = {}, temperature: float = 0.0, seed: int = 42) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # Populate the templates with custom values
        control.insert("increase", str(custom_values["increase"]), origin="user")
        control.insert("decrease", str(custom_values["decrease"]), origin="user")
        treatment.insert("increase", str(custom_values["increase"]), origin="user")
        treatment.insert("decrease", str(custom_values["decrease"]), origin="user")

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
            remarks=str({"increase": custom_values["increase"], "decrease": custom_values["decrease"]})
        )

        return test_case


class DispositionEffectMetric(RatioScaleMetric):
    """
    A metric that measures the presence and strength of the Disposition Effect based on a set of test results.

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results, flip_treatment=True)