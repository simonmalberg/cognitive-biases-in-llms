from abc import ABC, abstractmethod
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np
import re


class PopulationError(Exception):
    """A class for exceptions raised during the population of test cases."""
    def __init__(self, message: str, template: Template = None, model_output: str = None):
        extended_message = message
        if template is not None:
            try:
                extended_message += f"\n\n--- TEMPLATE ---\n{template}"
            except:
                pass
        if model_output is not None:
            extended_message += f"\n\n--- MODEL OUTPUT ---\n{model_output}"

        super().__init__(extended_message)   
        self.message = message
        self.template = template
        self.model_output = model_output


class DecisionError(Exception):
    """A class for exceptions raised during the decision of test cases."""
    pass


class MetricCalculationError(Exception):
    """A class for exceptions raised during the calculation of metric for a given bias."""
    pass


class LLM(ABC):
    """
    Abstract base class representing a Large Language Model (LLM) capable of generating and performing cognitive bias test cases.
    
    Attributes:
        NAME (str): The name of the model.
        shuffle_answer_options (bool): Whether or not answer options shall be randomly shuffled when making a decision.
    """

    def __init__(self, shuffle_answer_options: bool = False):
        self.NAME = "llm-abstract-base-class"
        self.shuffle_answer_options = shuffle_answer_options

    @abstractmethod
    def prompt(self, prompt: str, temperature: float = 0.0, seed: int = 42) -> str:
        """
        Prompts the LLM with a text input and returns the LLM's answer.

        Args:
            prompt (str): The input prompt text.
            temperature (float): The temperature value of the LLM.
            seed (int): The seed for controlling the LLM's output.

        Returns:
            str: The LLM's answer to the input prompt.
        """
        pass

    @abstractmethod
    def populate(self, control: Template, treatment: Template, scenario: str, temperature: float = 0.0, seed: int = 42) -> tuple[Template, Template]:
        """
        Populates given control and treatment templates based on the provided scenario.

        Args:
            control (Template): The control template that shall be populated.
            treatment (Template): The treatment template that shall be populated.
            scenario (str): A string describing the scenario/context for the population.
            temperature (float): The temperature value of the LLM.
            seed (int): The seed for controlling the LLM's output.

        Returns:
            tuple[Template, Template]: The populated control and treatment templates.
        """
        pass
    
    def _decide(self, template: Template, temperature: float = 0.7, seed: int = 42) -> tuple[str, str, int, list[str], list[int]]:
        """
        Prompts the model to choose one answer option from a decision-making task defined in the provided template.

        The decision is obtained through a two-step prompt: First, the model is presented with the decision-making test and can respond freely. Second, the model is instructed to extract the final answer from its previous response.

        Args:
            template (Template): The template defining the decision-making task.
            temperature (float): The temperature value of the LLM.
            seed (int): The seed for controlling the LLM's output.

        Returns:
            tuple[str, str, int, list[str], list[int]]: The raw model response, the model's extraction response, the number of the selected option (None if no selected option could be extracted), the answer option texts, and the order of answer options.
        """

        # 1. Load the decision and extraction prompts
        decision_prompt = self._PROMPTS['decision_prompt']
        extraction_prompt = self._PROMPTS['extraction_prompt']

        # 2A. Format the template and insert it into the decision prompt
        decision_prompt = decision_prompt.replace("{{test_case}}", template.format(shuffle_options=self.shuffle_answer_options, seed=seed))
        options, option_order = template.get_options(shuffle_options=self.shuffle_answer_options, seed=seed)

        # 2B. Obtain a response from the LLM
        try:
            decision_response = self.prompt(decision_prompt, temperature=temperature, seed=seed)
        except Exception as e:
            raise DecisionError(f"Could not obtain a decision from the model to the following prompt:\n\n{decision_prompt}\n\n{e}")

        # 3A. Insert the decision options and the decision response into the extraction prompt
        extraction_prompt = extraction_prompt.replace("{{options}}", "\n".join(f"Option {index}: {option}" for index, option in enumerate(options, start=1)))
        extraction_prompt = extraction_prompt.replace("{{answer}}", decision_response)

        # 3B. Let the LLM extract the final chosen option from its previous answer
        try:
            extraction_response = self.prompt(extraction_prompt, temperature=temperature, seed=seed)
        except Exception as e:
            raise DecisionError(f"An error occurred while trying to extract the chosen option with the following prompt:\n\n{extraction_prompt}\n\n{e}")

        # 3C. Extract the option number from the extraction response
        pattern = r'\b(?:[oO]ption) (\d+)\b'
        match = re.search(pattern, extraction_response)
        chosen_option = int(match.group(1)) if match else None

        if chosen_option is None:
            raise DecisionError(f"Could not extract the chosen option from the model's response:\n\n{decision_response}\n\n{extraction_response}\n\nNo option number detected in response.")

        return decision_response, extraction_response, chosen_option, options, option_order

    def decide(self, test_case: TestCase, temperature: float = 0.7, seed: int = 42) -> DecisionResult:
        """
        Makes the decisions defined in the provided test case (i.e., typically chooses one option from the control template and one option from the treatment template).

        Args:
            test_case (TestCase): The TestCase object defining the tests/decisions to be made.
            temperature (float): The temperature value of the LLM.
            seed (int): The seed for controlling the LLM's output.

        Returns:
            DecisionResult: A DecisionResult representing the decisions made by the LLM.
        """
        # Declare the results variables
        control_answer, control_extraction, control_option, control_option_texts, control_option_order = None, None, None, [], []
        treatment_answer, treatment_extraction, treatment_option, treatment_option_texts, treatment_option_order = None, None, None, [], []

        # Obtain decisions for the control and treatment decision-making tasks
        if test_case.CONTROL is not None:
            control_answer, control_extraction, control_option, control_option_texts, control_option_order = self._decide(test_case.CONTROL, temperature=temperature, seed=seed)
        if test_case.TREATMENT is not None:
            treatment_answer, treatment_extraction, treatment_option, treatment_option_texts, treatment_option_order = self._decide(test_case.TREATMENT, temperature=temperature, seed=seed)
        
        # Create a DecisionResult object containing the final decisions
        decision_result = DecisionResult(
            model=self.NAME,
            control_options=control_option_texts,
            control_option_order=control_option_order,
            control_answer=control_answer,
            control_decision=control_option,
            treatment_options=treatment_option_texts,
            treatment_option_order=treatment_option_order,
            treatment_answer=treatment_answer,
            treatment_decision=treatment_option,
            temperature=temperature,
            seed=seed
        )

        return decision_result
    
    def decide_all(self, test_cases: list[TestCase], temperature: float = 0.0, seed: int = 42, max_retries: int = 5) -> list[DecisionResult]:
        """
        Function to decide on all test cases in the list.
        
        Args:
            test_cases (list[TestCase]): A list of test cases to decide on.
            temperature (float): The temperature value of the LLM.
            seed (int): A seed for deterministic randomness
            max_retries (int): The maximum number of retries for a failed decision.
            
        Returns:
            list[DecisionResult]: A list of DecisionResult objects representing the decisions made by the LLM.
        """
        all_decisions = []
        for test_id, test_case in enumerate(test_cases):
            # try to make a decision for the test case within max_retries times
            for _ in range(max_retries):
                try:
                    test_decision = self.decide(test_case, temperature, seed)
                    seed += max_retries + 1
                    break
                except DecisionError as e:
                    test_decision = None
                    seed += 1
                    print(f"Warning: the model {self.NAME} failed to make a decision on the test case {test_id}.\nError: {e}\nRetrying...")
            # checking if the decision was successful in the end
            if test_decision is None:
                print(f"Max retries of {max_retries} reached for the test case {test_id} by the model {self.NAME}.\nSkipping...")
            all_decisions.append(test_decision)
            
        return all_decisions

    def _validate_population(self, template: Template, insertions: dict, raw_model_output: str = None) -> bool:
        """
        Validates if a model's generated insertions are valid for the provided template.

        Args:
            template (Template): The Template object for which insertions were generated.
            insertions (dict): A dictionary with all insertions that were generated by the model. Keys should be the patterns/gap instructions and values should be the generated insertions.
            raw_model_output (str): The raw model output. This is used for failure diagnostics in case the validation is unsuccessful.

        Returns:
            bool: True if the validation was successful. Otherwise, a PopulationError is raised.
        """

        # Get the remaining gaps from the template
        gaps = template.get_gaps()

        # Verify that insertions were generated for all remaining gaps
        for gap in gaps:
            if gap not in insertions:
                raise PopulationError(f"The gap '{gap}' has not been filled.", template, raw_model_output)

        # Verify that all generated insertions refer to gaps that exist
        for pattern in insertions.keys():
            if pattern not in gaps:
                raise PopulationError(f"An insertion was generated for a non-existing gap '{pattern}'.", template, raw_model_output)

        # Verify that all generated insertions are valid (i.e., not empty and not identical with the original gap instruction)
        for pattern in insertions.keys():
            if insertions[pattern] == None or insertions[pattern].strip() == "":
                raise PopulationError(f"Invalid insertion '{insertions[pattern]}' attempted into gap '{pattern}'. Insertion is empty.", template, raw_model_output)
            
            stripped_pattern = pattern.strip("[[").strip("]]").strip("{{").strip("}}")
            stripped_insertion = insertions[pattern].strip("[[").strip("]]").strip("{{").strip("}}")
            if stripped_insertion == stripped_pattern:
                raise PopulationError(f"Invalid insertion '{insertions[pattern]}' attempted into gap '{pattern}'. Insertion is identical with the gap instruction.", template, raw_model_output)

        return True


class TestGenerator(ABC):
    """
    Abstract base class for test generators. A test generator is responsible for generating test cases for a particular cognitive bias.
    
    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.    
    """

    def __init__(self):
        self.BIAS = "None"
    
    def sample_custom_values(self, num_instances: int = 5, iteration_seed: int = 42) -> dict:
        """
        Sample custom values for the test case generation.

        Args:
            num_instances (int): The number of instances expected to be generated for each scenario.
            iteration_seed (int): The seed to use for sampling the custom values.

        Returns:
            dict: A dictionary containing the sampled custom values.
        """
        return {}
    
    def select_custom_values_for_step(self, custom_values: dict, step: int) -> dict:
        """
        Selects the custom values for the given step from the dictionary of custom values.
        
        Args:
            custom_values (dict): A dictionary containing the custom values for all steps.
            step (int): The step for which to select the custom values.
            
        Returns:
            dict: A dictionary containing the custom values for the given step."""
        refined_dict = {}
        for key, values in custom_values.items():
            refined_dict[key] = values[step]
            
        return refined_dict
    
    def generate_all(
    self, model: LLM, scenarios: list[str], temperature: float = 0.0, seed: int = 42, num_instances: int = 5, max_retries: int = 5
) -> list[TestCase]:
        """
        Generate several test cases for each provided scenario.
        
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
        for scenario in scenarios:
            # creating a seed for each scenario, which fits the range of valid seeds for NumPy
            iteration_seed = hash(scenario + str(seed)) % (2**32)
            # get the custom values for the scenario
            sampled_values = self.sample_custom_values(num_instances, iteration_seed)
            for step in range(num_instances):
                # taking the subdictionary of custom values for the current step only
                sampled_values_step = self.select_custom_values_for_step(sampled_values, step)
                for _ in range(max_retries):
                    try:
                        test_case = self.generate(model, scenario, sampled_values_step, temperature, iteration_seed)
                        # if the test case is generated successfully, increment the seed to not collide with potential retries and break the retry loop
                        iteration_seed += max_retries + 1
                        break
                    except Exception as e:
                        test_case = None
                        iteration_seed += 1
                        print(
                                f"Warning: Generating the test case failed.\nScenario: {scenario}\nIteration seed: {iteration_seed}\nError: {e}\nRetrying..."
                            )
                # checking whether the generation is successful in the end. Otherwise, notify the user
                if test_case is None:
                    print(f"Max retries of {max_retries} reached for bias {self.BIAS}, scenario {scenario}.\nSkipping...")
                test_cases.append(test_case)
                
        return test_cases
    
    @abstractmethod
    def generate(self, model: LLM, scenario: str, config_values: dict = {}, temperature: float = 0.0, seed: int = 42) -> TestCase:
        """
        Generates a test case for the cognitive bias associated with this test generator.

        Args:
            model (LLM): The LLM model to use for generating the test case.
            scenario (str): The scenario for which to generate the test case.
            config_values (dict): A dictionary containing the configuration data for the test case.
            temperature (float): The temperature to use for generating the test case.
            seed (int): A seed for deterministic randomness.

        Returns:
            A TestCase object representing the generated test case.
        """
        pass

    def load_config(self, bias: str) -> TestConfig:
        """
        Loads the test configuration from the specified XML file.

        Args:
            path (str): The path to the XML file containing the test configuration.

        Returns:
            A TestConfig object representing the loaded test configuration.
        """
        return TestConfig(f"./biases/{bias.title().replace(' ', '')}/config.xml")

    def populate(self, model: LLM, control: Template, treatment: Template, scenario: str, temperature: float = 0.0, seed: int = 42) -> tuple[Template, Template]:
        """
        Populates the control and treatment templates using the provided LLM model and scenario.

        Args:
            model (LLM): The LLM model to use for populating the templates.
            control (Template): The control template.
            treatment (Template): The treatment template.
            scenario (str): The scenario for which to populate the templates.
            temperature (float): The temperature to use for generating the test cases.
            seed (int): A seed for deterministic randomness.

        Returns:
            A tuple containing the populated control and treatment templates.
        """

        # Populate the templates using the model and scenario
        control, treatment = model.populate(control, treatment, scenario, temperature, seed)

        return control, treatment

# TODO: This class is to be removed
class Metric(ABC):
    """
    Abstract base class for metrics. A metric is responsible for measuring the presence and strength of a cognitive bias in a Large Language Model (LLM).
    
    Attributes:
        BIAS (str): The cognitive bias associated with this metric.
    """

    def __init__(self):
        self.BIAS = "None"

    @abstractmethod
    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        pass
    
class AggregationMetric:
    """
    A metric that aggregates the evaluations of individual cognitive bias tests and computes a single bias metric value.
    
    𝔅 = (∑ wᵢ𝔅ᵢ) \ (∑ wᵢ)
    
    where: 
    - 𝔅ᵢ is bias of the individual test i
    - wᵢ is the weight of the individual test i (parameter). Default value is 1; for Loss Aversion, it is the test hyperparameter.
    
    Attributes:
        bias_results (np.array): The array of bias metric values for the individual tests.
        weights (np.array): The array of weights for the individual tests.
    """
    def __init__(self, bias_results: np.array, weights: np.array = np.array([1])):
        self.bias_results = bias_results
        self.weights = weights
    
    def compute(self) -> float:
        """
        Compute the aggregated metric value.
        
        Returns:
            float: The aggregated metric value.
        """
        return round(np.sum(self.weights * self.bias_results) / np.sum(self.weights), 2)
    

class RatioScaleMetric:
    """
    A metric that measures the presence and strength of a cognitive bias test equipped with a ratio scale.
    
    𝔅(â₁,â₂,x) = k ⋅ Δ(|Δ[â₁,x]|, |Δ[â₂,x]|) / max(|Δ[â1,x]|, |Δ[â₂,x]|) 
    
    where: 
    - â₁ and â₂ are the control and treatment answers, respectively
    - x is the test parameter (e.g., present in the Anchoring test and Hindsight Bias test)
    - k := ±1 (a constant factor)
    - Δ[â,x] := â - x
    
    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): A list of test results to be used for the metric calculation.
        k (np.array): The constant factor for the metric calculation.
        x (np.array): The test parameter.
        test_weights (np.array): The array of weights for the individual tests. Required for the metric aggregation.
        flip_control (bool): Whether to flip the control answer w.r.t. the centre of the scale.
        flip_treatment (bool): Whether to flip the treatment answer w.r.t. the centre of the scale.
    """
    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]], k: np.array = np.array([-1]), x: np.array = np.array([0]), test_weights: np.array = np.array([1]), flip_control: bool = False, flip_treatment: bool = False):
        self.test_results = test_results
        self.k = k
        self.x = x
        self.test_weights = np.repeat(test_weights, len(test_results))[:, None]
        self.flip_control = flip_control
        self.flip_treatment = flip_treatment
        
    def _compute(self, control_answer: np.array, treatment_answer: np.array) -> np.array:
        """
        Calculation of the ratio scale metric according to the formula above.
        
        Args:
            control_answer (np.array): The answer chosen in the control version.
            treatment_answer (np.array): The answer chosen in the treatment version.
            x (int): The test parameter.
        
        Returns:
            np.array: The metric value for each test case.
        """
        delta_control_abs, delta_treatment_abs = np.abs(control_answer - self.x), np.abs(treatment_answer - self.x)
        metric_value = self.k * (delta_control_abs - delta_treatment_abs) / (np.maximum(delta_control_abs, delta_treatment_abs) + 1e-8)
        
        return metric_value
    
    def compute(self) -> np.array:
        """
        Compute the ratio scale metric for the all provided tests.
        
        Returns:
            np.array: The metric value for each test case.
        """
        # make sure all pairs are not None
        self.test_results = [
            pair for pair in self.test_results if pair[0] is not None and pair[1] is not None
        ]
        try:
            # extract indices of the chosen answers
            control_answer = np.array(
                [
                    [decision_result.CONTROL_DECISION]
                    for (_, decision_result) in self.test_results
                ]
            )
            treatment_answer = np.array(
                [
                    [decision_result.TREATMENT_DECISION]
                    for (_, decision_result) in self.test_results
                ]
            )
            # if the control answer should be flipped
            if self.flip_control:
                # extract the length of the scale
                scale_length = len(self.test_results[0][1].CONTROL_OPTIONS)
                # flip the control answer
                control_answer = scale_length - 1 - control_answer
            # if the treatment answer should be flipped
            if self.flip_treatment:
                # extract the length of the scale
                scale_length = len(self.test_results[0][1].TREATMENT_OPTIONS)
                # flip the treatment answer
                treatment_answer = scale_length - 1 - treatment_answer
            # also account for the case when the control is not present in the test: e.g., Illusion of Control.
            # assume for these cases we have strictly odd number of options (central element is well-defined)
            if not control_answer.size:
                control_answer = np.array(len(self.test_results) * [[(len(self.test_results[0][1].CONTROL_OPTIONS) - 1) // 2]])
            biasedness_scores = self._compute(control_answer, treatment_answer)
        except Exception as e:
            print(e)
            raise MetricCalculationError(f"Error filtering test results: {e}")
        return biasedness_scores
    
    def aggregate(self, biasedness_scores: np.array) -> float:
        """
        Aggregate the ratio scale metric values for the all provided tests.
        
        Args:
            biasedness_scores (np.array): The metric value for each test case.
        
        Returns:
            float: The aggregated metric value.
        """
        return AggregationMetric(biasedness_scores, self.test_weights).compute()


class NominalScaleMetric:
    """
    A metric that measures the presence and strength of a cognitive bias test equipped with a nominal scale.
    
    𝔅(â₁,â₂) = (k ⋅ f(â₂ − â₁) + b) ⋅ (1 - 2|â₁ - x|)
    
    where: 
    - â₁ ∈ {0,1} and â₂ ∈ {0,1} are the control and treatment answers, respectively
    - x ∈ {0,1} is the test parameter (e.g., present in Bandwagon Effect test)
    - k := ±1, b := 0,1 (constant factors)
    - f(⋅) ∈ {|⋅|, id} (a function)
    
    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): A list of test results to be used for the metric calculation.
        options_labels (np.array): The array describing a map from options to labels {0,1}. Required to extract the type of the chosen answers.
        x (np.array): The test parameter.
        k (int): The constant for the metric calculation.
        b (int): The constant for the metric calculation.
        f (str): The function for the metric calculation.
        test_weights (np.array): The array of weights for the individual tests. Required for the metric aggregation.
    """
    def __init__(self, test_results: list[tuple[TestCase, DecisionResult]], options_labels: np.array = np.empty(0), x: np.array = np.empty(0), k: int = 1, b: int = 0, f: str = "id", test_weights: np.array = np.array([1])):
        self.test_results = test_results
        self.options_labels = options_labels
        self.x = x
        self.k = k
        self.b = b
        self.f = f
        self.test_weights = np.repeat(test_weights, len(test_results))[:, None]
        
    def _compute(self, control_answer: np.array, treatment_answer: np.array) -> np.array:
        """
        Calculation of the nominal scale metric according to the formula above.
        
        Args:
            control_answer (np.array): The answer chosen in the control version.
            treatment_answer (np.array): The answer chosen in the treatment version.
        
        Returns:
            np.array: The metric value for each test case.
        """
        factor = 1
        if np.any(self.x):
            factor -= 2 * np.abs(control_answer - self.x)
        if self.f == "abs":
            f = np.abs
        elif self.f == "id":
            f = lambda x: x
        else:
            raise MetricCalculationError(f"Unknown function '{self.f}' in the metric calculation.")
        
        return (self.k * f(treatment_answer - control_answer) + self.b) * factor
        
    def compute(self) -> np.array:
        """
        Compute the nominal scale metric for the all provided tests.
        
        Returns:
            np.array: The metric value for each test case.
        """
        try:
            # make sure all pairs are not None
            self.test_results = [
                pair
                for pair in self.test_results
                if pair[0] is not None and pair[1] is not None
            ]
            # extract chosen answers
            control_answer = np.array(
                [
                    decision_result.CONTROL_DECISION
                    for (_, decision_result) in self.test_results
                ]
            )
            treatment_answer = np.array(
                [
                    decision_result.TREATMENT_DECISION
                    for (_, decision_result) in self.test_results
                ]
            )
            # extract the type of the chosen answers
            # also account for the case when the control is not present in the test
            if np.any(control_answer):
                control_answer = self.options_labels[control_answer]
            else:
                control_answer = np.array([0])
            treatment_answer = self.options_labels[treatment_answer]
            # compute the biasedness scores
            biasedness_scores = np.mean(self._compute(control_answer, treatment_answer))
        except Exception as e:
            print(e)
            raise MetricCalculationError(f"Error computing the metric: {e}")
        return biasedness_scores
    
    def aggregate(self, biasedness_scores: np.array) -> float:
        """
        Aggregate the nominal scale metric values for the all provided tests.
        
        Args:
            biasedness_scores (np.array): The metric value for each test case.
        
        Returns:
            float: The aggregated metric value.
        """
        return AggregationMetric(biasedness_scores, self.test_weights).compute()
