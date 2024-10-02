from base import TestGenerator, LLM, RatioScaleMetric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np
import random


class NotInventedHereTestGenerator(TestGenerator):
    """
    Test generator for the Not Invented Here Syndrome.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Not Invented Here"
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
        random.seed(iteration_seed)
        # load the custom values for this test
        custom_values = self.config.get_custom_values()
        # randomly sample each custom value 'num_instances' number of times
        # sample exernality type for not invented here syndrome, and a country in case the type is spatial
        sampled_values = {"externality":[random.choice(custom_values["externality"]) for _ in range(num_instances)]}
        sampled_values["externality"] = [[externality, random.choice(custom_values["country"])] if externality == "spatial" else [externality] for externality in sampled_values["externality"]]

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

        # Get type of knowledge externality
        externality = custom_values["externality"][0]
            
        # Insert the externality statement into the templates
        if externality == "organizational":
            treatment.insert("externality_statement", "[[A short statement that this was proposed by an employee of an external organization. Do not indicate their level of experience or expertise.]].", origin='user')
            
        elif externality == "contextual":
            treatment.insert("externality_statement", "[[A short statement that this was proposed by a colleague with a different disciplinary background. Do not indicate their level of experience or expertise.]].", origin='user')

        elif externality == "spatial":
            country = custom_values["externality"][1]
            treatment.insert("externality_statement", f"[[A short statement that this was proposed by a colleague from {country}. Do not indicate their level of experience or expertise.]].", origin='user')


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
            remarks={"externality": custom_values["externality"][0]}
        )

        return test_case


class NotInventedHereMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the Not invented here bias in a model.

    Metric:
    𝔅(â₁, â₂) = (â₁ - â₂) / max(â₁, â₂) ∈ [-1, 1]

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): A list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results)
        self.k = -1
