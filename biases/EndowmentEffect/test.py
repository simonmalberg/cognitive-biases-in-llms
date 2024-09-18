from base import TestGenerator, LLM, Metric, MetricCalculationError
from tests import TestCase, Template, TestConfig, DecisionResult
from tqdm import tqdm
import numpy as np
import random


class EndowmentEffectTestGenerator(TestGenerator):
    """
    Test generator for the Endowment Effect.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Endowment Effect"
        self.config: TestConfig = super().load_config(self.BIAS)
        
    def generate_all(
        self, model: LLM, scenarios: list[str], temperature: float = 0.0, seed: int = 42, num_instances: int = 5, max_retries: int = 5
    ) -> list[TestCase]:
        """
        Generate several test cases for each provided scenarios.
        
        Args:
            model (LLM): The LLM to use for generating the test cases.
            scenarios (list[str]): A list of scenarios to generate the test cases for.
            temperature (float): The temperature to use for generating the test cases.
            seed (int): The seed to use for generating the test cases.
            num_instances (int): The number of instances to generate for each scenario.
            max_retries (int): The maximum number of retries in generation of all tests for this bias.
        """
        test_cases: list[TestCase] = []
        sampled_values: dict = {}
        num_retries = 0
        for scenario in scenarios:
            # creating a seed for each scenario
            iteration_seed = hash(scenario + str(seed))
            random.seed(iteration_seed)
            # load the custom values for this test
            custom_values = self.config.get_custom_values()
            # randomly sample each custom value 'num_instances' number of times
            # in this case, it is an empty dictionary as there are no custom values
            sampled_values = {}
            for step in range(num_instances):
                try:
                    test_case = self.generate(model, scenario, sampled_values, step, temperature, iteration_seed)
                    test_cases.append(test_case)
                except Exception as e:
                    num_retries += 1
                    print(
                            f"Generating the test case failed.\nScenario: {scenario}\nIteration seed: {iteration_seed}\nError: {e}"
                        )
                iteration_seed += 1
                # checking that the generation has not failed too many times for the given scenario
                if num_retries > max_retries:
                    print(f"Max retries reached for bias {self.BIAS}, scenario {scenario}, and seed {seed}")
                    break
                
        return test_cases

    def generate(
        self, model: LLM, scenario: str, custom_values: dict = {}, step: int = 0, temperature: float = 0.0, seed: int = 42
    ) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()
        # Populate the templates using the model and the scenario
        control, treatment = super().populate(model, control, treatment, scenario, temperature, seed)

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


class EndowmentEffectMetric(Metric):
    """
    Metric calculator for the Endowment Effect.

    Metric:
    𝔅 = (â₂ - â₁) / a

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    a = â₁ - â (if â₂ - â₁ < 0) or else a = ã - â₁, where ã is the maximum option, â - the minimum option.

    """

    def _compute(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        max_option: np.array,
    ) -> np.array:
        """
        Compute the metric for the Bandwagon effect.

        Args:
            control_answer (np.array): The answer chosen in the control version.
            treatment_answer (np.array): The answer chosen in the treatment version.
            max_option (np.array): The maximum answer option.
            min_option (np.array): The minimum answer option (= 0).

        Returns:
            np.array: The metric value for the test case.
        """
        delta = control_answer - treatment_answer
        biasedness = delta / (
            (delta >= 0) * control_answer
            + (delta < 0) * (max_option - control_answer)
            + 10e-8
        )

        return biasedness

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        try:
            # make sure all pairs are not None
            test_results = [
                pair for pair in test_results if pair[0] is not None and pair[1] is not None
            ]
            # extract the max option
            max_option = len(test_results[0][1].CONTROL_OPTIONS) - 1
            # extract answers
            control_answer = np.array(
                [
                    [decision_result.CONTROL_DECISION]
                    for (_, decision_result) in test_results
                ]
            )
            treatment_answer = np.array(
                [
                    [decision_result.TREATMENT_DECISION]
                    for (_, decision_result) in test_results
                ]
            )
            biasedness_scores = np.mean(
                self._compute(control_answer, treatment_answer, max_option)
            )
        except Exception as e:
            print(e)
            raise MetricCalculationError(f"Error computing the metric: {e}")
        return round(biasedness_scores, 2)
