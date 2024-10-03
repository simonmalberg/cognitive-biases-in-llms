from base import TestGenerator, LLM, RatioScaleMetric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np
import re


class HindsightBiasTestGenerator(TestGenerator):
    """
    Test generator for the Hindsight Bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Hindsight Bias"
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
        sampled_values = {}
        np.random.seed(iteration_seed)
        # load the custom values for this test
        custom_values = self.config.get_custom_values()
        for key, value in custom_values.items():
            if key == "percentage":
                sampled_values[key] = getattr(np.random, value[0])(
                    int(value[1]), int(value[2]), size=num_instances
                )
        
        return sampled_values

    def generate(
        self,
        model: LLM,
        scenario: str,
        custom_values: dict = {},
        temperature: float = 0.0,
        seed: int = 42,
    ) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # Populate the treatment template with a custom value
        treatment.insert("percentage", str(int(custom_values['percentage'])), origin="user")

        # Populate the templates using the model and the scenario
        control, treatment = super().populate(
            model, control, treatment, scenario, temperature, seed
        )

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


class HindsightBiasMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the Hindsight bias in a model.

    Metric:
    𝔅 = (‖ â₁ − a' ‖₁ − ‖ â₂ − a' ‖₁) / max[‖ â₁ − a' ‖₁, ‖ â₂ − a' ‖₁] ∈ [-1, 1];

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    a' is the option closest to the ground truth percentage (sampled using custom values);
    
    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """
    
    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results)
        # extract the options closest to the ground truth values and set them as parameters x_1 and x_2.
        self.x_1 = [
            [
                insertion.text
                for insertion in test_case.TREATMENT.get_insertions()
                if insertion.pattern == "percentage"
            ]
            for (test_case, _) in test_results
        ]
        self.x_1 = np.array(
            [[round(int(x[0]) / 10) + 5] for x in self.x_1]
        )
        self.x_2 = self.x_1
        self.k = 1
