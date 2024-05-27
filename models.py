from abc import ABC, abstractmethod
from tests import Template
import re
import random
import openai
import json


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
    def populate(self, control: Template, treatment: Template, scenario: str) -> tuple[Template, Template]:
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

    # TODO: discuss how we track that control and treatment options are populated identically
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

    def decide(self):
        pass


class GptThreePointFiveTurbo(LLM):
    """
    A class representing a GPT-3.5-Turbo LLM that populates test cases according to the scenario.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(self):
        super().__init__()
        self.NAME = "gpt-3.5-turbo"

    def _fill_gaps(self, text, replacements):
        for placeholder, replacement in replacements.items():
            text = text.replace(f"[[{placeholder}]]", replacement)
        return text

    def populate(self, control: Template, treatment: Template, scenario: str) -> tuple[Template, Template]:
        # Define the prompt to the LLM
        prompt = f"""\
You will be given a scenario, a control template, and a treatment template. \
The templates have gaps indicated by double square brackets containing instructions on how to fill them, e.g., [[Fill in the blanks]]. \
Some gaps can be present in both the control and treatment templates.

--- SCENARIO ---
{scenario}

--- CONTROL TEMPLATE ---
{control.format(insert_headings=True, show_type=False, show_generated=True)}

--- TREATMENT TEMPLATE ---
{treatment.format(insert_headings=True, show_type=False, show_generated=True)}

Please provide the texts to fill in the gaps in the following JSON format:
{{
    "placeholder1": "replacement1",
    "placeholder2": "replacement2",
    ...
}}

Fill in the gaps according to the instructions and scenario.\
"""

        # Obtain a response from the LLM
        response = openai.ChatCompletion.create(
            model=self.NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        # Parse the replacements proposed by the LLM
        replacements = json.loads(response.choices[0].message['content'].strip())

        # Insert the proposed replacements into the template gaps
        filled_control_text = self._fill_gaps(control.serialize(), replacements)
        filled_treatment_text = self._fill_gaps(treatment.serialize(), replacements)

        # Create new templates with the gaps filled
        filled_control = Template(filled_control_text)
        filled_treatment = Template(filled_treatment_text)

        return filled_control, filled_treatment

    def decide(self):
        pass