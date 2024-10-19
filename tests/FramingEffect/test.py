from core.base import TestGenerator, LLM, RatioScaleMetric, MetricCalculationError
from core.testing import TestCase, Template, TestConfig, DecisionResult
import numpy as np


class FramingEffectTestGenerator(TestGenerator):
    """
    Test generator for the Framing Effect.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Framing Effect"
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
        np.random.seed(iteration_seed)
        # load the custom values for this test
        custom_values = self.config.get_custom_values()
        # randomly sample each custom value 'num_instances' number of times
        # in this case, we are sampling the first_percentage value from randint from the provided range
        sampled_values = {
            key: getattr(np.random, value[0])(int(value[1]), int(value[2]), size=num_instances)
            for key, value in custom_values.items() if key == "first_percentage"
        }

        return sampled_values
    

    def generate(
        self, model: LLM, scenario: str, custom_values: dict = {}, temperature: float = 0.0, seed: int = 42
    ) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # Populate the templates with the custom values
        control.insert("first_percentage", str(custom_values["first_percentage"]), origin='user')
        treatment.insert("second_percentage", str(100 - custom_values["first_percentage"]), origin='user')

        # Populate the templates using the model and the scenario
        control, treatment = super().populate(model, control, treatment, scenario, temperature, seed)

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
            remarks=None,
        )

        return test_case


class FramingEffectMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the Framing effect in a model.

    Metric:
    𝔅(â₁, â₂) = (â₂ - â₁) / max(â₁, â₂) ∈ [-1, 1]

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): A list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results)
