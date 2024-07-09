from abc import ABC, abstractmethod
from tests import TestCase, Template, TestConfig, DecisionResult
import yaml
import random


class LLM(ABC):
    """
    Abstract base class representing a Large Language Model (LLM) capable of generating and performing cognitive bias test cases.
    
    Attributes:
        NAME (str): The name of the model.
        PROMPTS (dict): A dictionary containing the prompts used to interact with the model.
    """

    def __init__(self):
        self.NAME = "llm-abstract-base-class"
        with open("prompts.yml") as prompts:
            self.PROMPTS = yaml.safe_load(prompts)
         
    def shuffle_options(self, template: Template, seed: int = 42) -> tuple[Template, Template]:
        """
        Function to shuffle the order of the answer options in the given template.
        
        Args:
            template (Template): The template in which to shuffle the options.
            seed (int): A seed for deterministic randomness.
        
        Returns:
            A tuple containing the shuffled template and the dict with the shuffled options.
        """
        if not template:
            return None, None

        options = {}
        random.seed(seed)

        # Extract the options from the template
        template_idx, option_elements = list(
            zip(
                *[
                    (idx, element)
                    for idx, element in enumerate(template.elements)
                    if element[1] == "option"
                ]
            )
        )
        option_texts = [element[0] for element in option_elements]

        # option_idx contains the indices of the options in the template, we want indices of actual options (from 1 to n)
        option_idx = list(range(len(template_idx)))
        random.shuffle(option_idx)

        # Replace the options in the template with the shuffled ones and save the order of the options in the DesicionResult instance (key+1 since the options are enumerated from 1)
        for i, (i_template, i_option) in enumerate(zip(template_idx, option_idx)):
            options[i+1] = option_texts[i_option]
            template.elements[i_template] = (option_texts[i_option], "option")
        
        return template, options

    @abstractmethod
    def populate(self, control: Template, treatment: Template, scenario: str) -> tuple[Template, Template]:
        pass

    @abstractmethod
    def decide(self, test_case: TestCase) -> DecisionResult:
        pass


class TestGenerator(ABC):
    """
    Abstract base class for test generators. A test generator is responsible for generating test cases for a particular cognitive bias.
    
    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.    
    """

    def __init__(self):
        self.BIAS = "None"

    @abstractmethod
    def generate_all(self, model: LLM, scenarios: list[str], config_values: dict = {}, seed: int = 42) -> list[TestCase]:
        """
        Generates all test cases at once for the cognitive bias associated with this test generator.

        Args:
            model (LLM): The LLM model to use for generating the test case.
            scenario (list[str]): The list of scenarios for which to generate the test case.
            config_values (dict): A dictionary containing the configuration data for the test case from the respective XML file.
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
        return TestConfig(f"./biases/{bias.replace(' ', '')}/config.xml")

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