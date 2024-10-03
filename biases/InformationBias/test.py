from base import TestGenerator, LLM, RatioScaleMetric
from tests import TestCase, DecisionResult
import numpy as np
import random


class InformationBiasTestGenerator(TestGenerator):
    """
    Test generator for Information Bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Information Bias.
    """

    def __init__(self):
        self.BIAS = "InformationBias"
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
        np.random.seed(iteration_seed)

        # initialize dictionary to store sampled values
        sampled_values = {"prior":[], "posterior_high":[], "control_posterior_low":[], "treatment_posterior_low":[]}

        # load the custom values for this test
        custom_values = self.config.get_custom_values()

        # randomly sample each custom value 'num_instances' number of times
        for _ in range(num_instances):

            # Sample prior confidence in course of action
            min, max, step = custom_values["prior"]
            prior = random.choice(np.arange(int(min), int(max)+1, int(step)))
            sampled_values["prior"].append(prior)

            # Sample posteriors lower and higher than prior
            min, max, step = custom_values["posterior_high"]
            posterior_high = random.choice(np.arange(int(prior)+5, int(max)+1, int(step)))
            sampled_values["posterior_high"].append(posterior_high)

            min, max, step = custom_values["control_posterior_low"]
            control_posterior_low = random.choice(np.arange(int(min), int(max)+1, int(step)))
            sampled_values["control_posterior_low"].append(control_posterior_low)

            min, max, step = custom_values["treatment_posterior_low"]
            treatment_posterior_low = random.choice(np.arange(int(min), int(prior), int(step)))
            sampled_values["treatment_posterior_low"].append(treatment_posterior_low)

        return sampled_values

    def generate(
        self,
        model: LLM,
        scenario: str,
        custom_values: dict = {},
        temperature: float = 0.0,
        seed: int = 42,
    ) -> TestCase:

        # Load the control and treatment template
        control = self.config.get_control_template()
        treatment = self.config.get_treatment_template()

         # Retrieve and format sampled custom values
        prior = str(custom_values["prior"])+ "%"
        posterior_high = str(custom_values["posterior_high"])+ "%"
        control_posterior_low = str(custom_values["control_posterior_low"])+ "%"
        treatment_posterior_low = str(custom_values["treatment_posterior_low"])+ "%"

        # Insert the sampled values into the control template
        control.insert('prior', prior, origin='user')
        control.insert('posterior_high', posterior_high, origin='user')
        control.insert('control_posterior_low', control_posterior_low, origin='user')

        # Insert the sampled values into the treatment template
        treatment.insert('prior', prior, origin='user')
        treatment.insert('posterior_high', posterior_high, origin='user')
        treatment.insert('treatment_posterior_low', treatment_posterior_low, origin='user')

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
            remarks=None
        )

        return test_case


class InformationBiasMetric(RatioScaleMetric):

    """
    A class that describes the quantitative evaluation of the Information Bias in a model.

    Metric:
    𝔅(â₁, â₂) = (â₁ - â₂) / max(â₁, â₂) ∈ [-1, 1]

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively (control is shifted by 1: â₁ := â₁ + 1).;

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): A list of test results to be used for the metric calculation.
    """
    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):

        max_option = len(test_results[0][1].CONTROL_OPTIONS)
        
        for idx, _ in enumerate(test_results):

            test_results[idx][1].CONTROL_DECISION = max_option - test_results[idx][1].CONTROL_DECISION
            test_results[idx][1].TREATMENT_DECISION = max_option - test_results[idx][1].TREATMENT_DECISION
            test_results[idx][1].TREATMENT_DECISION -= 1
            
        super().__init__(test_results)
        self.k = 1
        