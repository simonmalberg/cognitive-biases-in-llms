from tests import Template, TestCase, DecisionResult
from base import LLM
import re
import random
from openai import OpenAI
import json
import yaml
import warnings


class PopulationError(Exception):
    """A class for exceptions raised during the population of test cases."""
    pass


class DecisionError(Exception):
    """A class for exceptions raised during the decision of test cases."""
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

    def decide(self, test_case: TestCase) -> DecisionResult:
        pass

'''
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
        # TODO be able to handle empty templates (e.g., control = None)

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

        # TODO enable OpenAI's JSON mode

        # Obtain a response from the LLM
        response = self.client.chat.completions.create(
            model=self.NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        # Parse the replacements proposed by the LLM
        replacements = json.loads(response.choices[0].message['content'].strip())

        # TODO validation of replacements

        # Insert the proposed replacements into the template gaps
        filled_control_text = self._fill_gaps(control.serialize(), replacements)
        filled_treatment_text = self._fill_gaps(treatment.serialize(), replacements)

        # Create new templates with the gaps filled
        filled_control = Template(filled_control_text)
        filled_treatment = Template(filled_treatment_text)

        return filled_control, filled_treatment

    def decide(self):
        pass
'''

    
class GptThreePointFiveTurbo(LLM):
    """
    A class representing a GPT-3.5-Turbo LLM that populates test cases according to the scenario 
    starting from the brackets that are either identical for both control and treatment or unique for control, 
    and then adding those unique for treatment.

    Attributes:
        NAME (str): The name of the model.
        CLIENT (OpenAI): The OpenAI client used to interact with the GPT-3.5-Turbo model.
        PROMPTS (dict): A dictionary containing the prompts used to interact with the model.
    """

    def __init__(self):
        super().__init__()
        self.NAME = "gpt-3.5-turbo"
        self.client = OpenAI()

    def generate_misc(self, prompt: str) -> str:
        """
        Utility function to generate a miscellaneous prompt to the LLM.
        """
        response = self.client.chat.completions.create(
            model=self.NAME,
            messages=[{"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def populate_intersection(self, control: Template, treatment: Template, scenario: str) -> tuple[Template, Template]:
        # Load the prompt to the LLM
        prompt = self.PROMPTS['first_prompt']
        system_content = self.PROMPTS['system_prompt']
        # Insert the scenario and control template into the prompt
        prompt = prompt.replace("{{scenario}}", scenario)
        prompt = prompt.replace("{{control_template}}", control.format(insert_headings=True,
                                                                       show_type=False, 
                                                                       show_generated=True))
        # Obtain a response from the LLM
        response = self.client.chat.completions.create(
            model=self.NAME,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
            )
        # Parse the replacements proposed by the LLM
        replacements = json.loads(response.choices[0].message.content)
        self.validate_population(control, replacements)
        # Insert the proposed replacements into the template gaps
        control.insert_generated_values(replacements)
        treatment.insert_generated_values(replacements)
        
        return control, treatment
    

    def validate_population(self, template: Template, replacements: dict):
            # test 1: check that number of replacements is equal to the number of placeholders in the template
            template_entries = re.findall(r'\[\[(.*?)\]\]', template.serialize())
            if len(replacements) < len(template_entries):
                raise PopulationError("Not enough replacements generated.")
            # check that all placeholders are filled, and the replacements are not empty / same as the initial placeholders
            for ((placeholder, replacement), entry) in zip(replacements.items(), template_entries):
                if not placeholder == "[[" + entry + "]]":
                    raise PopulationError("A placeholder was skipped/altered.") 
                if not replacement:
                    raise PopulationError("A placeholder was not filled.")
                if replacement == entry:
                    raise PopulationError("Generated passage is the same as the placeholder.")


    def populate_difference(self, treatment: Template) -> Template:
        # Load the prompt to the LLM
        prompt = self.PROMPTS['second_prompt']
        system_content = self.PROMPTS['system_prompt']
        # Insert the treatment template into the prompt
        prompt = prompt.replace("{{treatment_template}}", treatment.format(insert_headings=True,
                                                                         show_type=False, 
                                                                         show_generated=True))
        # Obtain a response from the LLM
        response = self.client.chat.completions.create(
            model=self.NAME,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
            )
        # Parse the replacements proposed by the LLM
        replacements = json.loads(response.choices[0].message.content)
        self.validate_population(treatment, replacements) 
        treatment.insert_generated_values(replacements)

        return treatment


    def populate(self, control: Template, treatment: Template, scenario: str) -> tuple[Template, Template]:
        try:
            if not control:
                control = treatment
                treatment, control = self.populate_intersection(control, treatment, scenario)
                control = None
            else:
                control, treatment = self.populate_intersection(control, treatment, scenario)
                # Check if there are any placeholders left in the treatment template
                if re.findall(r'\[\[(.*?)\]\]', treatment.serialize()):
                    treatment = self.populate_difference(treatment)
        except PopulationError as e:
            warnings.warn(e)
            return None, None
        
        return control, treatment
    
    def decide(self, test_case: TestCase) -> DecisionResult:
        try:
            pass
        except DecisionError as e:
            warnings.warn(e)
            return None, None
