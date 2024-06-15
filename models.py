from abc import ABC, abstractmethod
from tests import Template, TestCase
import re
import random
from openai import OpenAI
import json
import yaml
import warnings
import lmql
import tiktoken

class PopulationError(Exception):
    """A class for exceptions raised during the population of test cases."""
    pass

class DecisionError(Exception):
    """A class for exceptions raised during the decision of test cases."""
    pass


class DecisionResult:
    """
    A class representing the result of a decision made by an LLM for a specific test case.
    """

    def __init__(self, control_option: int, treatment_option: int, confidences: list, explanation: str):
        # TODO chosen option, confidence in each option, (explanation)
        self.CONTROL_OPTION = control_option
        self.TREATMENT_OPTION = treatment_option
        self.CONFIDENCES = None
        self.EXPLANATION = None

    def __str__(self) -> str:
        return f'---DecisionResult---\n\nCONTROL OPTION: {self.CONTROL_OPTION}\nTREATMENT OPTION: {self.TREATMENT_OPTION}\n\n------'

    def __repr__(self) -> str:
        return self.__str__()



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

    @abstractmethod
    def populate(self, control: Template, treatment: Template, scenario: str) -> tuple[Template, Template, dict]:
        pass

    @abstractmethod
    def decide(self, test_case: TestCase) -> DecisionResult:
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

    def populate_intersection(self, control: Template, treatment: Template, scenario: str) -> tuple[Template, Template, dict]:
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


    def populate_difference(self, treatment: Template) -> tuple[Template, dict]:
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
                treatment, control, replacements = self.populate_intersection(control, treatment, scenario)
                control = None
            else:
                control, treatment, replacements = self.populate_intersection(control, treatment, scenario)
                # Check if there are any placeholders left in the treatment template
                if re.findall(r'\[\[(.*?)\]\]', treatment.serialize()):
                    treatment, augm_dict = self.populate_difference(treatment)
                    replacements.update(augm_dict)
        except PopulationError as e:
            warnings.warn(e)
            return None, None, None
        
        return control, treatment, replacements
    
    # The ChatGPT model does not support the constraint on the answer, hence rewrote the functions. 
    # Did not delete the code since it should work with other models.
    #
    # Two possible implementations: 
    #
    # @lmql.query
    # def get_decision(prompt_case):
    #     '''lmql
    #     "{promp_case}. The selected option is: [ANSWER: int]"
    #     return ANSWER
    #     '''
    # 
    # def get_decision(self, prompt_case: str) -> int:
    #     answer = lmql.F("{promp_case}. The selected option is: [ANSWER: int]")
    #     return answer(prompt_case, model=lmql.model(self.NAME), decoder=self.DECODER)
    
    
    # functions using the OpenAI API (naive extraction of the digits in the answer, 
    # since the model does not support constraints on the answer + logit_bias does not guarantee the exclusion of other options)
    #
    # def extract_numerical(self, answer: str) -> int:
        # extract all numbers from the answer and return their concatenation
        # return int(''.join(re.findall(r'.*(\d+):.*', answer)))
    
    def get_decision(self, prompt_case: str) -> int:
        # Get the tokeniser corresponding to a given OpenAI model
        # enc = tiktoken.encoding_for_model(self.NAME)
        # Extract answer options from the prompt
        # answer_options = re.findall(r'Option (\d+): .*', prompt_case)
        # logit_bias_map = {}
        # for option in answer_options:
            # making sure that the model has a high bias towards the options in the output (obviously problematic)
            # logit_bias_map[str(enc.encode(option)[0])] = 100
        completions = self.client.chat.completions.create(
            model=self.NAME,
            messages=[{"role": "user", "content": prompt_case}],
            # logit_bias=logit_bias_map
        )
        chosen_option = completions.choices[0].message.content
        # chosen_option = self.extract_numerical(chosen_option)
        
        return chosen_option 
    
    def decide(self, test_case: TestCase) -> DecisionResult:
        try:
            prompt = self.PROMPTS['decision_prompt']
            test_prompt = prompt.replace("{{test_case}}", test_case.TREATMENT.format(insert_headings=True,
                                                                                     show_type=False, 
                                                                                     show_generated=True))
            treatment_option = self.get_decision(test_prompt)
            if not test_case.CONTROL:
                return DecisionResult(None, treatment_option, None, None)
            else:
                test_prompt = prompt.replace("{{test_case}}", test_case.CONTROL.format(insert_headings=True,
                                                                                       show_type=False, 
                                                                                       show_generated=True))
                control_option = self.get_decision(test_prompt)
                return DecisionResult(control_option, treatment_option, None, None)
        except DecisionError as e:
            warnings.warn(e)
            return DecisionResult(None, None, None)
