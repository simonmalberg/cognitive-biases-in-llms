from abc import ABC, abstractmethod
import re
import random


class DecisionResult:
    """
    A class representing the result of a decision made by an LLM for a specific test case.
    """

    def __init__(self, model):
        pass


class LLM(ABC):
    """
    Abstract base class representing a Large Language Model (LLM) capable of generating and performing cognitive bias test cases.
    
    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(self):
        self.NAME = "llm-abstract-base-class"

    @abstractmethod
    def populate(self, control: str, treatment: str, scenario: str):
        pass

    @abstractmethod
    def decide(self) -> DecisionResult:
        pass


class RandomModel(LLM):
    """
    A class representing a random model that populates test cases with random samples of words from the scenario. This class is implemented for testing purposes only.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(self):
        self.NAME = "random-model"

    def populate(self, control: str, treatment: str, scenario: str):
        # Replace any text between brackets with a random sample of 1-4 words from the scenario
        def replace_with_sample(match):
            sampled_words = ' '.join(random.sample(scenario.split(), random.randint(1, 4)))
            return f"[[{sampled_words}]]"
    
        control = re.sub(r'\[\[(.*?)\]\]', replace_with_sample, control)
        treatment = re.sub(r'\[\[(.*?)\]\]', replace_with_sample, treatment)
        
        return control, treatment

    def decide(self):
        pass


class GptThreePointFiveTurbo(LLM):

    def __init__(self):
        pass

    def populate(self, control: str, treatment: str, scenario: str):
        pass

    def decide(self):
        pass