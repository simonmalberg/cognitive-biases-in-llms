from base import TestGenerator, LLM, RatioScaleMetric
from tests import TestCase, Template, DecisionResult
import numpy as np


class LossAversionTestGenerator(TestGenerator):
    """
    Test generator for the Loss Aversion bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Loss Aversion bias.
    """

    def __init__(self):
        self.BIAS = "Loss Aversion"
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
        # in this case, we are sampling the base_amount, lambda_coef, and the control/treatment choices' order
        index = np.random.choice(
                range(len(custom_values["control_choice"])), size=num_instances
            )
        for key, value in custom_values.items():
            if key == "control_choice":
                sampled_values["control_choice"] = [
                    value[index[n]] for n in range(num_instances)
                ]
                sampled_values["other_control_choice"] = [
                    value[1 - index[n]] for n in range(num_instances)
                ]
            elif key == "treatment_choice":
                sampled_values["treatment_choice"] = [
                    value[index[n]] for n in range(num_instances)
                ]
                sampled_values["other_treatment_choice"] = [
                    value[1 - index[n]] for n in range(num_instances)
                ]
            else:
                sampled_values[key] = getattr(np.random, value[0])(
                    float(value[1]), float(value[2]), size=num_instances
                )
        # calculating the lambda amount
        sampled_values["lambda_amount"] = np.round(
            sampled_values["lambda_coef"] * sampled_values["base_amount"], 1
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
        # Load the templates
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()
        # Insert the custom values into the templates
        control.insert("control_choice", custom_values['control_choice'], origin="user")
        control.insert("other_control_choice", custom_values['other_control_choice'], origin="user")
        treatment.insert("treatment_choice", custom_values['treatment_choice'], origin="user")
        treatment.insert("other_treatment_choice", custom_values['other_treatment_choice'], origin="user")
        for template in [control, treatment]:
            template.insert("lambda_amount", str(custom_values['lambda_amount']), origin="user")
            template.insert("base_amount", str(custom_values['base_amount']), origin="user")

        # Populate the template using the model and the scenario
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


class LossAversionMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the Loss aversion bias in a model.

    Individual metric:
    𝔅(â₁, â₂) = k ⋅ (â₁ - â₂) / max(â₁, â₂) ∈ [-1, 1]

    Batch metric:
    𝔅 = (∑ wᵢ𝔅ᵢ) / (∑ wᵢ) ∈ [-1, 1]

    where:
    â₂ is the chosen answer for the i-th test;
    â₁ is the fixed central (neutral) option of the scale;
    k is the parameter that reflects the order of choices in the test case (k = 1 if the guaranteed choice is presented first, k = -1 otherwise).
    wᵢ is the loss aversion hyperparameter in the i-th test (test_weights). Set as 1.
    """
    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results)
        # set the coefficient in the metric: it depends on the 'index' custom value that we sampled
        # (and reflects which scheme is presented first, i.e., which scheme is A)
        self.k = [
            [
                insertion.text
                for insertion in test_case.TREATMENT.get_insertions()
                if insertion.pattern == "treatment_choice"
            ]
            for (test_case, _) in self.test_results
        ]
        self.k = np.array([[-1] if "guarantees" in k[0] else [1] for k in self.k])
        # we also need to flip treatment options
        self.flip_treatment = True
        # extract lambda parameters from the test cases and set them as the test_weights in the metric
        # lambda_amounts = np.array([
        #     [
        #         float(insertion.text)
        #         for insertion in test_case.TREATMENT.get_insertions()
        #         if insertion.pattern == "lambda_amount"
        #     ]
        #     for (test_case, _) in self.test_results
        # ])
        # base_amounts = np.array([
        #     [
        #         float(insertion.text)
        #         for insertion in test_case.TREATMENT.get_insertions()
        #         if insertion.pattern == "base_amount"
        #     ]
        #     for (test_case, _) in self.test_results
        # ])
        # self.test_weights = lambda_amounts / base_amounts
