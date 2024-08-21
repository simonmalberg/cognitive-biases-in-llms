from base import TestGenerator, LLM, Metric, PopulationError, MetricCalculationError
from tests import TestCase, Template, TestConfig, DecisionResult
import re
import numpy as np


class AnchoringBiasTestGenerator(TestGenerator):
    """
    Test generator for the Anchoring Bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Anchoring Bias.
    """

    def __init__(self):
        self.BIAS: str = "Anchoring Bias"
        self.config: TestConfig = super().load_config(self.BIAS)

    def _custom_population(
        self, model: LLM, completed_template: Template, custom_values: dict, seed: int
    ) -> None:
        """
        Custom population method for the Anchoring Bias test case.

        Args:
            model (LLM): The LLM model to use for generating the anchor sentence.
            completed_template (Template): The assembled template with scenario for the test case.
            custom_values (dict): The custom values for the test case.
            seed (int): The seed.
        """
        # Loading the anchor sentence generation prompt
        anchor_prompt = custom_values["anchor_sentence"]
        # generate the anchor sentence
        anchor_sentence = model.prompt(anchor_prompt, seed=seed)
        # Inserting the anchor sentence into the (treatment) template
        completed_template.insert("anchor_sentence", anchor_sentence)
        # Explicitly extract the numerical value from the generated anchor sentence
        anchor = re.findall(r"\d+", anchor_sentence)
        if len(anchor) != 1:
            raise PopulationError(
                "The anchor sentence should contain exactly one numerical value"
            )

    def generate_all(
        self, model: LLM, scenarios: list[str], seed: int = 42
    ) -> list[TestCase]:
        # Load the custom values from the test configuration
        custom_values = self.config.get_custom_values()

        # Create test cases for all scenarios
        test_cases: list[TestCase] = []
        for scenario in scenarios:
            try:
                test_case = self.generate(model, scenario, custom_values, seed)
                test_cases.append(test_case)
            except Exception as e:
                print(
                    f"Generating the test case failed.\nScenario: {scenario}\nSeed: {seed}"
                )
                print(e)

        return test_cases

    def generate(
        self, model: LLM, scenario: str, custom_values: dict = {}, seed: int = 42
    ) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # Populate the treatment template with the anchor sentence
        self._custom_population(model, treatment, custom_values, seed)
        # Get dictionary of inserted values
        treatment_values = treatment.inserted_values

        # Populate the templates using the model and the scenario
        control, treatment = super().populate(model, control, treatment, scenario)

        # Create a test case object
        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=treatment,
            generator=model.NAME,
            scenario=scenario,
            control_values=None,
            treatment_values=treatment_values,
            variant=None,
            remarks=None,
        )

        return test_case


class AnchoringBiasMetric(Metric):
    """
    A class that describes the quantitative evaluation of the anchoring bias in a model.

    Metric:
    𝔅 = (‖ â₁ − a' ‖₁ − ‖ â₂ − a' ‖₁) / a ∈ [-1, 1];

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    a' is the answer option closest to the anchor value;
    a = ‖ â₁ − a' ‖₁ (if numerator is positive) or else a = ‖ â₂ − a' ‖₁.

    """

    def _compute(
        self,
        control_answer: np.array,
        treatment_answer: np.array,
        answer_options: np.array,
        anchor: np.array,
    ) -> np.array:
        """
        Compute the metric for the anchoring bias.

        Args:
            control_answer (np.array, shape (batch, 1)): The answer(s) chosen for the control variant.
            treatment_answer (np.array, shape (batch, 1)): The answer(s) chosen for the treatment variant.
            answer_options (np.array, shape (batch, num_options)): The answer options available in the test case.
            anchor (np.array, shape (batch, 1)): The anchor value(s) used in the test case.

        Returns:
            The anchoring bias metric value.
        """
        delta = np.abs(control_answer - anchor) - np.abs(treatment_answer - anchor)
        closest_idx = np.argmin(np.abs(anchor - answer_options), axis=1, keepdims=True)
        closest_val = np.take_along_axis(answer_options, closest_idx, axis=1)
        result = delta / (
            (delta >= 0) * np.abs(control_answer - closest_val)
            + (delta < 0) * np.abs(treatment_answer - closest_val)
            + 10e-8
        )

        return result

    def assemble_options(self, options_list: list[dict]) -> np.array:
        """
        Assemble the answer options into a numpy array.

        Args:
            options (dict): The answer options for the test case.

        Returns:
            np.array: The assembled answer options array.
        """
        answer_options = np.array([])
        for options in options_list:
            numerical_options = [int(re.findall(r"-?\d+\.?\d*", s)[0]) for s in options]
            if not answer_options.size:
                answer_options = np.array([numerical_options])
            else:
                answer_options = np.vstack((answer_options, numerical_options))

        return answer_options

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        try:
            # make sure all pairs are not None
            test_results = [
                pair
                for pair in test_results
                if pair[0] is not None and pair[1] is not None
            ]
            # extract answer options from the test results
            answer_options = self.assemble_options(
                [
                    decision_result.CONTROL_OPTIONS
                    for (_, decision_result) in test_results
                ]
            )
            # extract indices of the chosen answers (-1 because the option indices are 1-indexed)
            control_answer_idx = (
                np.array(
                    [
                        [decision_result.CONTROL_DECISION]
                        for (_, decision_result) in test_results
                    ]
                )
                - 1
            )
            treatment_answer_idx = (
                np.array(
                    [
                        [decision_result.TREATMENT_DECISION]
                        for (_, decision_result) in test_results
                    ]
                )
                - 1
            )
            # extract the anchor values
            anchor_values = np.array(
                [
                    [
                        int(
                            re.findall(
                                r"\d+", test_case.TREATMENT_VALUES["anchor_sentence"][0]
                            )[0]
                        )
                        for test_case, _ in test_results
                    ]
                ]
            )
            # extract the chosen answers (-1 because the option indices are 1-indexed)
            control_answer = np.take_along_axis(
                answer_options, control_answer_idx, axis=1
            )
            treatment_answer = np.take_along_axis(
                answer_options, treatment_answer_idx, axis=1
            )
            biasedness_scores = np.mean(
                self._compute(
                    control_answer, treatment_answer, answer_options, anchor_values
                )
            )
        except Exception as e:
            print(e)
            raise MetricCalculationError("The metric could not be computed.")
        return np.around(biasedness_scores, 2)
