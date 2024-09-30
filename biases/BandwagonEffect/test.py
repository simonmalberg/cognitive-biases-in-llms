from base import TestGenerator, LLM, RatioScaleMetric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np


class BandwagonEffectTestGenerator(TestGenerator):
    """
    Test generator for the Bandwagon Effect.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Bandwagon Effect"
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
        # in this case, we are sampling the majority opinion
        index = np.random.choice(
                range(len(custom_values["majority_opinion"])), size=num_instances
            )
        for key, value in custom_values.items():
            if key == "majority_opinion":
                sampled_values["majority_opinion"] = [
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

        # Populate the templates with custom values
        for template in [control, treatment]:
            template.insert("majority_opinion", custom_values["majority_opinion"], origin="user")

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


class BandwagonEffectMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the Bandwagon Effect in a model.

    Metric:
    𝔅(â₁, â₂) = k ⋅ (â₁ - â₂) / max(â₁, â₂) ∈ [-1, 1]
    where:
    â₂, â₁ are the chosen answers for the treatment and control versions, respectively.
    k is the parameter that reflects the majority opinion in the test case (k = -1 if it is A, k = 1 otherwise).

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results)
        # set the coefficient in the metric: it depends on the 'index' custom value that we sampled
        # (and reflects which opinion is presented as the majority one)
        self.k = [
            [
                insertion.text
                for insertion in test_case.CONTROL.get_insertions()
                if insertion.pattern == "majority_opinion"
            ]
            for (test_case, _) in self.test_results
        ]
        self.k = np.array([[-1] if "A" in k[0] else [1] for k in self.k])
        # we flip the treatment answers
        self.flip_treatment = True
