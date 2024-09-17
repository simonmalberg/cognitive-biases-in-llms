from base import TestGenerator, LLM, Metric, MetricCalculationError
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np
from tqdm import tqdm
import random


class FramingEffectTestGenerator(TestGenerator):
    """
    Test generator for the Framing Effect.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    """

    def __init__(self):
        self.BIAS: str = "Framing Effect"
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
        for scenario in tqdm(scenarios):
            # creating a seed for each scenario
            iteration_seed = hash(scenario + str(seed))
            random.seed(iteration_seed)
            # load the custom values for this test
            custom_values = self.config.get_custom_values()
            # randomly sample each custom value 'num_instances' number of times
            # in this case, we are sampling the first_percentage value from randint from the provided range
            sampled_values = {
                key: [random.randint(float(value[0]), float(value[1])) for _ in range(num_instances)]
                for key, value in custom_values.items() if key == "first_percentage"
            }
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
            # checking that the generation has not failed too many times for the given bias
            if num_retries > max_retries:
                print(f"Max retries reached for bias {self.BIAS}, seed {seed}")
                break
                
        return test_cases

    def generate(
        self, model: LLM, scenario: str, custom_values: dict = {}, step: int = 0, temperature: float = 0.0, seed: int = 42
    ) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # Populate the templates with the custom values sampled in the generate_all method
        # We retrive the value that was generated for the current step
        first_percentage = custom_values["first_percentage"][step]
        control.insert("first_percentage", str(first_percentage), origin='user')
        treatment.insert("second_percentage", str(100 - first_percentage), origin='user')

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


class FramingEffectMetric(Metric):
    """
    A class that describes the quantitative evaluation of the framing effect in a model.

    Metric:
    𝔅 = (â₂ - â₁) / a

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively (where Bad is 4, ..., Great is 0);
    a = ã - â₁ (if â₂ - â₁ > 0) or else a = â₁ - â, where ã is the maximum option (0-indexed), â - the minimum option (0).

    """

    def _compute(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        max_option: np.array,
        min_option: np.array,
    ) -> np.array:
        """
        Compute the metric for the Framing Effect.

        Args:
            control_answer (np.array): The answer chosen in the control version.
            treatment_answer (np.array): The answer chosen in the treatment version.
            max_option (np.array): The maximum answer option.
            min_option (np.array): The minimum answer option.

        Returns:
            np.array: The metric value for the test case.
        """
        delta = treatment_answer - control_answer
        metric_value = delta / (
            (delta >= 0) * (max_option - control_answer)
            + (delta < 0) * (control_answer - min_option)
            + 10e-8
        )

        return metric_value

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        try:
            # make sure all pairs are not None
            test_results = [
                pair
                for pair in test_results
                if pair[0] is not None and pair[1] is not None
            ]
            # extract the answer options' length
            len_answer_options = len(test_results[0][1].CONTROL_OPTIONS)
            min_option, max_option = 0, len_answer_options - 1
            # extract original unshuffled indices of the chosen answers
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
                self._compute(
                    control_answer, treatment_answer, max_option, min_option
                )
            )
        except Exception as e:
            print(e)
            raise MetricCalculationError(f"Error computing the metric: {e}")
        return round(biasedness_scores, 2)
