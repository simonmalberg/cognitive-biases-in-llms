from base import TestGenerator, LLM, RatioScaleMetric
from tests import TestCase, Template, DecisionResult
import numpy as np


class OptimismBiasTestGenerator(TestGenerator):
    """
    Test generator for the Optimism bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Optimism bias.
    """

    def __init__(self):
        self.BIAS = "Optimism Bias"
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
        # in this case, we are sampling the event_kind
        index = np.random.choice(
                range(len(custom_values["event_kind"])), size=num_instances
            )
        for key, value in custom_values.items():
            if key == "event_kind":
                sampled_values[key] = [
                    value[index[n]] for n in range(num_instances)
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
            template.insert("event_kind", custom_values['event_kind'], origin="user")

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


class OptimismBiasMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the optimism bias in a model.

    Individual metric:
    𝔅(â₁, â₂) = k ⋅ (â₁ - â₂) / max(â₁, â₂) ∈ [-1, 1]

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    k is the kind of event (-1: positive or 1: negative).
    
    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results)
        # set the coefficient in the metric: it depends on the 'index' custom value that we sampled
        # (and reflects which event kind is used in the test case)
        self.k = [
            [
                insertion.text
                for insertion in test_case.TREATMENT.get_insertions()
                if insertion.pattern == "event_kind"
            ]
            for (test_case, _) in self.test_results
        ]
        self.k = np.array([[-1] if "positive" in k[0] else [1] for k in self.k])
