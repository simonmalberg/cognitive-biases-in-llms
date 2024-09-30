from base import TestGenerator, LLM, RatioScaleMetric
from tests import TestCase, Template, DecisionResult
import numpy as np


class HaloEffectTestGenerator(TestGenerator):
    """
    Test generator for the Halo Effect bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Halo Effect bias.
    """

    def __init__(self):
        self.BIAS = "Halo Effect"
        self.config = super().load_config(self.BIAS)
        
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
        # randomly sample each custom value 'num_instances' number of times
        # in this case, we are sampling the type of perception
        for key, value in custom_values.items():
            if key == "perception":
                sampled_values[key] = [
                    value[n] for n in range(num_instances)
                ]
        
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

        # Insert the custom values into the template
        for template in [control, treatment]:
            template.insert("perception", custom_values['perception'], origin="user")

        # Populate the templates using the model and the scenario
        # first populate the treatment template, then the control template
        treatment, control = super().populate(
            model, treatment, control, scenario, temperature, seed
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


class HaloEffectMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the Halo effect in a model.

    Metric:
    𝔅(â₁, â₂) = k ⋅ (â₁ - â₂) / max(â₁, â₂) ∈ [-1, 1]
    where:
    â₂, â₁ are the chosen answers for the treatment and control versions, respectively.
    k is the parameter that reflects the type of halo (k = 1 for a positive one, k = -1 otherwise).

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results)
        self.k = [
                [
                    insertion.text
                    for insertion in test_case.TREATMENT.get_insertions()
                    if insertion.pattern == "perception"
                ]
                for (test_case, _) in test_results
            ]
        self.k = np.array([[1] if k == ["positively"] else [-1] for k in self.k])
