from base import TestGenerator, LLM, RatioScaleMetric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np
import random


class RiskCompensationTestGenerator(TestGenerator):
    """
    Test generator for the Risk Compensation.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Risk Compensation"
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

        sampled_values = {"initial_risk":[], "risk_reduction":[]}

        for _ in range(num_instances):
        
            # Sample prior confidence in course of action
            min, max, step = custom_values["initial_risk"]
            initial_risk = random.choice(np.arange(int(min), int(max)+1, int(step)))
            sampled_values["initial_risk"].append(initial_risk)

            # Sample posteriors lower and higher than prior
            min, max, step = custom_values["risk_reduction"]
            risk_reduction = random.choice(np.arange(int(min), int(initial_risk)-10+1, int(step)))
            sampled_values["risk_reduction"].append(risk_reduction)

        return sampled_values
    

    def generate(
        self,
        model: LLM,
        scenario: str,
        custom_values: dict = {},
        temperature: float = 0.0,
        seed: int = 42,
    ) -> TestCase:
        
        # retrieve the custom values
        initial_risk = custom_values["initial_risk"]
        risk_reduction = custom_values["risk_reduction"]

        # Load the control and treatment templates
        control: Template = self.config.get_control_template()      
        treatment: Template = self.config.get_treatment_template()  

        # Insert the sampled values into the control template
        control.insert('initial_risk', str(initial_risk)+"%", origin='user')

        # Insert the sampled values into the treatment template
        treatment.insert('initial_risk', str(initial_risk)+"%", origin='user')
        treatment.insert('risk_reduction', str(risk_reduction)+"%", origin='user')

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


class RiskCompensationMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of Risk Compensation in a model.

    Metric:
    𝔅(â₁, â₂) = (â₁ - â₂) / max(â₁, â₂) ∈ [-1, 1]

    where:
    â₁, â₂ are the chosen answers for the treatment and control versions, respectively;

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): A list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results)
        self.k = -1
