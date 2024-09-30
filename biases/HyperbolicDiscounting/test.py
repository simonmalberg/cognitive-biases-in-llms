from base import TestGenerator, LLM, RatioScaleMetric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np


class HyperbolicDiscountingTestGenerator(TestGenerator):
    """
    Test generator for the Hyperbolic Discounting.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Hyperbolic Discounting"
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
        # in this case, we are sampling the earlier_reward, later_coef, months_delay and the schemes' order
        index = int(
            np.random.choice(
                range(len(custom_values["scheme_control"])), size=num_instances
            )
        )
        for key, value in custom_values.items():
            if key == "scheme_control":
                sampled_values["control_scheme"] = [
                    value[index] for _ in range(num_instances)
                ]
                sampled_values["other_control_scheme"] = [
                    value[1 - index] for _ in range(num_instances)
                ]
            elif key == "scheme_treatment":
                sampled_values["treatment_scheme"] = [
                    value[index] for _ in range(num_instances)
                ]
                sampled_values["other_treatment_scheme"] = [
                    value[1 - index] for _ in range(num_instances)
                ]
            else:
                sampled_values[key] = getattr(np.random, value[0])(
                    int(value[1]), int(value[2]), size=num_instances
                )
        # Making sure the later reward is always less than 100
        sampled_values["later_reward"] = np.minimum(
            sampled_values["earlier_reward"] * np.array(sampled_values["later_coef"]),
            100,
        ).astype(int)

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
            template.insert(
                "control_scheme", custom_values["control_scheme"], origin="user"
            )
            template.insert(
                "other_control_scheme",
                custom_values["other_control_scheme"],
                origin="user",
            )
            template.insert(
                "treatment_scheme", custom_values["treatment_scheme"], origin="user"
            )
            template.insert(
                "other_treatment_scheme",
                custom_values["other_treatment_scheme"],
                origin="user",
            )
            template.insert(
                "earlier_reward", str(custom_values["earlier_reward"]), origin="user"
            )
            template.insert(
                "later_reward", str(custom_values["later_reward"]), origin="user"
            )
            template.insert(
                "months_delay", str(custom_values["months_delay"]), origin="user"
            )

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


class HyperbolicDiscountingMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the Hyperbolic Discounting in a model.

    Metric:
    𝔅(â₁, â₂) = k ⋅ (â₂ - â₁) / max(â₁, â₂) ∈ [-1, 1]
    where:
    â₂, â₁ are the chosen answers for the treatment and control versions, respectively.
    k is the parameter that reflects the order of schemes in the test case (k = -1 if an immediate reward is presented first, k = 1 otherwise).

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]]):
        super().__init__(test_results)
        # set the coefficient in the metric: it depends on the 'index' custom value that we sampled
        # (and reflects which scheme is presented first, i.e., which scheme is A)
        self.k = [
            [
                insertion.text
                for insertion in test_case.CONTROL.get_insertions()
                if insertion.pattern == "control_scheme"
            ]
            for (test_case, _) in self.test_results
        ]
        self.k = np.array([[-1] if "immediately" in k[0] else [1] for k in self.k])
