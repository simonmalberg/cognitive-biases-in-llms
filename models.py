from tests import Template, TestCase, DecisionResult
from base import LLM
import re
import random
from openai import OpenAI
import json
import warnings

def options_to_list(options: str) -> list:
    """
    Function to convert the string with selected options into a list of options.
    """
    try:
        return [int(x) for x in re.findall(r"Option (\d*)", options)]
    except ValueError:
        raise DecisionError("The decision could not be extracted.")

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

class GptThreePointFiveTurbo(LLM):
    """
    A class representing a GPT-3.5-Turbo LLM that populates test cases according to the scenario 
    starting from the brackets that are either identical for both control and treatment or unique for control, 
    and then adding those unique for treatment.

    Attributes:
        NAME (str): The name of the model.
        DECODER (str): The decoding method used to generate completions from the model.
        CLIENT (OpenAI): The OpenAI client used to interact with the GPT-3.5-Turbo model.
        PROMPTS (dict): A dictionary containing the prompts used to interact with the model.
    """

    def __init__(self):
        super().__init__()
        self.NAME = "gpt-3.5-turbo"
        self.DECODER = "argmax"
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

    def validate_population(self, template: Template, replacements: dict):
        # test 1: check that number of replacements is equal to the number of placeholders in the template
        template_entries = re.findall(r"\[\[(.*?)\]\]", template.serialize())
        if len(replacements) < len(template_entries):
            raise PopulationError("Not enough replacements generated.")
        # check that all placeholders are filled, and the replacements are not empty / same as the initial placeholders
        for (placeholder, replacement), entry in zip(
            replacements.items(), template_entries
        ):
            if not placeholder == "[[" + entry + "]]":
                raise PopulationError("A placeholder was skipped/altered.")
            if not replacement:
                raise PopulationError("A placeholder was not filled.")
            if replacement == entry:
                raise PopulationError(
                    "Generated passage is the same as the placeholder."
                )
                
    # TODO: make one function only (e.g., _populate) and keep track of inserted values in the Template class

    def populate_control(
        self, control: Template, treatment: Template, scenario: str
    ) -> tuple[Template, Template, dict]:
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

        return control, treatment, replacements

    def populate_treatment(self, treatment: Template) -> tuple[Template, dict]:
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

        return treatment, replacements

    def populate(self, control: Template, treatment: Template, scenario: str) -> tuple[Template, Template, dict]:
        try:
            if not control:
                control = treatment
                treatment, control, replacements = self.populate_control(control, treatment, scenario)
                control = None
            else:
                # TODO: make "control = self.populate_control(control, scenario)" and store all the replacements inside the Template class
                control, treatment, replacements = self.populate_control(control, treatment, scenario)
                # Check if there are any placeholders left in the treatment template
                if re.findall(r'\[\[(.*?)\]\]', treatment.serialize()):
                    treatment, augm_dict = self.populate_treatment(treatment)
                    replacements.update(augm_dict)
        except Exception as e:
            raise PopulationError(e)

        return control, treatment, replacements

    def decide(self, test_case: TestCase, seed: int = 42) -> DecisionResult:
        test_case.CONTROL, control_options, test_case.TREATMENT, treatment_options = (
            self.shuffle_options(test_case.CONTROL, test_case.TREATMENT, seed)
        )
        try:
            # load the test prompt + extraction prompt
            prompt, extraction_prompt = (
                self.PROMPTS["decision_prompt"],
                self.PROMPTS["extraction_prompt"],
            )
            test_prompt = prompt.replace(
                "{{test_case}}",
                test_case.TREATMENT.format(
                    insert_headings=True, show_type=False, show_generated=True
                ),
            )
            # get answer for treatment part
            treatment_answer = self.generate_misc(test_prompt)
            # put the answer into the extraction prompt
            extraction_prompt_treatment = extraction_prompt.replace(
                "{{answer}}", treatment_answer
            )
            # extract the selected decision(s) from the treatment part into a list
            treatment_decision = options_to_list(
                self.generate_misc(extraction_prompt_treatment)
            )
            if not test_case.CONTROL:
                return DecisionResult(
                    control_options,
                    None,
                    treatment_options,
                    treatment_decision,
                    None,
                    None,
                )
            else:
                test_prompt = prompt.replace(
                    "{{test_case}}",
                    test_case.CONTROL.format(
                        insert_headings=True, show_type=False, show_generated=True
                    ),
                )
                # get answer for control part
                control_answer = self.generate_misc(test_prompt)
                # put the answer into the extraction prompt
                extraction_prompt_control = extraction_prompt.replace(
                    "{{answer}}", control_answer
                )
                # extract the selected option(s) from the control part into a list
                control_decision = options_to_list(
                    self.generate_misc(extraction_prompt_control)
                )
                return DecisionResult(
                    control_options,
                    control_decision,
                    treatment_options,
                    treatment_decision,
                    None,
                    None,
                )
        except Exception as e:
            raise DecisionError(e)
