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
         
    def shuffle_options(
        self, control: Template, treatment: Template, seed: int
    ) -> tuple[Template, Template]:
        """
        Function to shuffle the order of the options in the control and treatment templates.
        """
        random.seed(seed)
        control_options, treatment_options = None, None
        for template in [control, treatment]:
            if template:
                # extracting the options from the template
                option_idx, option_elements = list(
                    zip(
                        *[
                            (idx, element)
                            for idx, element in enumerate(template.elements)
                            if element[1] == "option"
                        ]
                    )
                )
                # TODO: shuffle not the options themselves, but their order
                option_texts = [element[0] for element in option_elements]
                random.shuffle(option_texts)
                # saving the order of the options for the DesicionResult instance
                if template == control:
                    # k+1 since the options are enumerated from 0 (and in the Template class from 1)
                    control_options = {k + 1: v for k, v in enumerate(option_texts)}
                else:
                    treatment_options = {k + 1: v for k, v in enumerate(option_texts)}
                # replacing the options in the template with the shuffled ones
                for idx, shuffled in zip(option_idx, option_texts):
                    template.elements[idx] = (shuffled, "option")

        return control, control_options, treatment, treatment_options

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

    def populate(self, model: LLM, control: Template, treatment: Template, scenario: str) -> tuple[Template, Template, dict]:
        """
        Populates the control and treatment templates using the provided LLM model and scenario.

        Args:
            model (LLM): The LLM model to use for populating the templates.
            control (Template): The control template.
            treatment (Template): The treatment template.
            scenario (str): The scenario for which to populate the templates.

        Returns:
            A tuple containing the populated control and treatment templates + replacements dict.
        """

        # Populate the templates using the model and scenario
        control, treatment, replacements = model.populate(control, treatment, scenario)

        return control, treatment, replacements


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