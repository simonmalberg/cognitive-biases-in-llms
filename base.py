from abc import ABC, abstractmethod
from tests import TestCase, Template, TestConfig, DecisionResult
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

    @abstractmethod
    def generate_all(self, model: LLM, scenarios: list[str], seed: int = 42) -> list[TestCase]:
        """
        Generates all test cases at once for the cognitive bias associated with this test generator.

        Args:
            model (LLM): The LLM model to use for generating the test case.
            scenarios (list[str]): The list of scenarios for which to generate the test case.
            seed (int): A seed for deterministic randomness.

        Returns:
            A list of TestCase objects representing the generated test cases.
        """
        pass
    
    @abstractmethod
    def generate(self, model: LLM, scenario: str, config_values: dict = {}, seed: int = 42) -> TestCase:
        """
        Generates a test case for the cognitive bias associated with this test generator.

        Args:
            model (LLM): The LLM model to use for generating the test case.
            scenario (str): The scenario for which to generate the test case.
            config_values (dict): A dictionary containing the configuration data for the test case.
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

    def populate(self, model: LLM, control: Template, treatment: Template, scenario: str) -> tuple[Template, Template]:
        """
        Populates the control and treatment templates using the provided LLM model and scenario.

        Args:
            model (LLM): The LLM model to use for populating the templates.
            control (Template): The control template.
            treatment (Template): The treatment template.
            scenario (str): The scenario for which to populate the templates.

        Returns:
            A tuple containing the populated control and treatment templates.
        """

        # Populate the templates using the model and scenario
        control, treatment = model.populate(control, treatment, scenario)

        return control, treatment


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