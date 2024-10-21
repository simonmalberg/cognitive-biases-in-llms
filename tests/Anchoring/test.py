from core.base import TestGenerator, LLM, RatioScaleMetric
from core.testing import TestCase, Template, TestConfig, DecisionResult
import numpy as np


class AnchoringTestGenerator(TestGenerator):
    """
    Test generator for the Anchoring.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Anchoring.
    """

    def __init__(self):
        self.BIAS: str = "Anchoring"
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
        # randomly sample each custom value 'num_instances' number of times
        # in this case, we are sampling the anchor value
        for key, value in custom_values.items():
            if key == "anchor":
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

        # Inserting the sample into the treatment template
        treatment.insert("anchor", str(int(custom_values['anchor'])), origin="user")
        
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


class AnchoringMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the anchoring in a model.
    
    Metric:
    𝔅 = (‖ â₁ − a' ‖₁ − ‖ â₂ − a' ‖₁) / max[‖ â₁ − a' ‖₁, ‖ â₂ − a' ‖₁] ∈ [-1, 1];

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    a' is the answer option closest to the anchor value;
    """
    
    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results)
        # set the coefficient in the metric
        self.k = 1
        # set the anchor values as the parameters x_1 and x_2 in the metric
        self.x_1 = [
                [
                    insertion.text
                    for insertion in test_case.TREATMENT.get_insertions()
                    if insertion.pattern == "anchor"
                ]
                for (test_case, _) in self.test_results
            ]
        self.x_1 = np.array([[round(int(anchor[0]) / 10)] for anchor in self.x_1])
        self.x_2 = self.x_1
