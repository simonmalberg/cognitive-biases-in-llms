from tests import Template, TestCase, DecisionResult
from base import LLM
import re
import random


class RandomModel(LLM):
    """
    A class representing a random model that populates test cases with random samples of words from the scenario. This class is implemented for testing purposes only.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(self):
        self.NAME = "random-model"

    def prompt(self, prompt: str) -> str:
        return "RANDOM RESPONSE"

    def populate(self, control: Template, treatment: Template, scenario: str) -> tuple[Template, Template]:
        # Serialize the templates into strings
        control_str = control.serialize()
        treatment_str = treatment.serialize()

        # Replace any text between brackets with a random sample of 1-4 words from the scenario
        def replace_with_sample(match):
            sampled_words = ' '.join(random.sample(scenario.split(), random.randint(1, 4)))
            return f"[[{sampled_words}]]"
    
        control_str = re.sub(r'\[\[(.*?)\]\]', replace_with_sample, control_str)
        treatment_str = re.sub(r'\[\[(.*?)\]\]', replace_with_sample, treatment_str)

        # Deserialize the populated strings back into templates
        control, treatment = Template(control_str), Template(treatment_str)
        
        return control, treatment

    def decide(self, test_case: TestCase) -> DecisionResult:
        pass