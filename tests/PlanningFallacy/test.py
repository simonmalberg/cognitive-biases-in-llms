from core.base import TestGenerator, LLM, RatioScaleMetric
from core.testing import TestCase, Template, TestConfig, DecisionResult
import numpy as np

class PlanningFallacyTestGenerator(TestGenerator):
    """
    Test generator for the Planning Fallacy.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Planning Fallacy"
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
            if key == "estimation_update":
                sampled_values[key] = 10 * getattr(np.random, value[0])(
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
        treatment.insert("estimation_update", str(int(custom_values['estimation_update'])), origin="user")

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


class PlanningFallacyMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the Planning fallacy in a model.

    Metric:
    𝔅(â₁, â₂) = (â₁ + x₁ - â₂) / max(â₁ + x₁, â₂) ∈ [-1, 1]

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    x₁ is the parameter that corresponds to the rational estimation update.

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): A list of test results to be used for the metric calculation.
    """
    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results)
        # extract the estimation updates' values and set them as the parameter x_1.
        self.x_1 = [
            [
                insertion.text
                for insertion in test_case.TREATMENT.get_insertions()
                if insertion.pattern == "estimation_update"
            ]
            for (test_case, _) in test_results
        ]
        self.x_1 = np.array(
            [[int(x[0]) // 10] for x in self.x_1]
        )
        # account for the sign of the parameter x_1 in the metric
        self.x_1 = -self.x_1
        # to make the estimator unbiased, we set the parameter x_2 to -𝔼[x_1] = -3
        self.x_2 = -3
        self.k = 1
