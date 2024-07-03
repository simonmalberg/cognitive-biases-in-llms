from abc import ABC, abstractmethod
from tests import TestCase, Template, TestConfig, DecisionResult


class LLM(ABC):
    """
    Abstract base class representing a Large Language Model (LLM) capable of generating and performing cognitive bias test cases.
    
    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(self):
        self.NAME = "llm-abstract-base-class"

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
    def generate_all(self, model: LLM, scenarios: list[str], config_values: dict, seed: int) -> list[TestCase]:
        pass
    
    @abstractmethod
    def generate(self, model: LLM, scenario: str, config_values: dict, seed: int) -> TestCase: 
        """
        Generates a test case for the cognitive bias associated with this test generator.

        Args:
            model (LLM): The LLM model to use for generating the test case.
            bias_dict (dict): A dictionary containing the bias data for the test case from the respective YAML file.
            scenario (str): The scenario for which to generate the test case.

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