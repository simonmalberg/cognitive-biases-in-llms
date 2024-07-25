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

class GPT(LLM):
    """
    An abstract class representing a GPT-based LLM from OpenAI API that populates test cases according to the scenario
    starting from the brackets that are either identical for both control and treatment or unique for control, 
    and then adding those unique for treatment.
    """

    def __init__(self):
        super().__init__()
        self.NAME = None
        self.DECODER = "argmax"
        self.client = OpenAI()

    def generate_misc(self, prompt: str) -> str:
        """
        Utility function to generate a miscellaneous prompt to the LLM.
        """
        response = self.client.chat.completions.create(
            model=self.NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def _validate_populate(self, template: Template, replacements: dict):
        # TODO Add function documentation

        # 1. Verify that number of replacements is equal to the number of placeholders in the template
        template_entries = re.findall(r"\[\[(.*?)\]\]", template.serialize())
        if len(replacements) < len(template_entries):
            raise PopulationError("Not enough replacements generated.")

        # 2. Verify that all placeholders are filled, and the replacements are not empty / same as the initial placeholders
        for (placeholder, replacement), entry in zip(replacements.items(), template_entries):
            if not placeholder == "[[" + entry + "]]":
                raise PopulationError("A placeholder was skipped/altered.")
            if not replacement:
                raise PopulationError("A placeholder was not filled.")
            if replacement == entry:
                raise PopulationError("Generated passage is the same as the placeholder.")
                
    def _populate(self, template: Template, scenario: str, kind: str) -> Template:
        """
        Function to populate a given template according to the scenario.
        
        Args:
            template (Template): The template to populate.
            scenario (str): The respective scenario.
            kind (str): The type of the template (control or treatment).
        """
        # Load the prompt to the LLM
        prompt = self.PROMPTS[f'{kind}_prompt']
        system_content = self.PROMPTS['system_prompt']
        # Insert the scenario and control template into the prompt
        prompt = prompt.replace("{{scenario}}", scenario)
        prompt = prompt.replace("{{" + f"{kind}_template" + "}}", template.format(insert_headings=True,
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
        self._validate_populate(template, replacements)
        template.insert_values(list(replacements.items()), kind='LLM')
        return template

    def populate(self, control: Template, treatment: Template, scenario: str) -> tuple[Template, Template]:
        try:
            if not control:
                control = None
            else:
                # 1. Populate the gaps in the control template based on the scenario
                control = self._populate(control, scenario, 'control')

                # 2. Fill the treatment template with the values that are shared with the control template
                control_values = {pattern: value[0] for pattern, value in control.inserted_values.items()}
                treatment.insert_values(list(control_values.items()), kind='LLM')

            # 3. Populate the gaps in the treatment template based on the scenario, if there are any placeholders left unfilled
            if re.findall(r'\[\[(.*?)\]\]', treatment.serialize()):
                treatment = self._populate(treatment, scenario, 'treatment')
        except Exception as e:
            raise PopulationError(e)

        return control, treatment

    def decide(self, test_case: TestCase, seed: int = 42) -> DecisionResult:
        test_case.CONTROL, control_options = self.shuffle_options(test_case.CONTROL, seed)
        test_case.TREATMENT, treatment_options = self.shuffle_options(test_case.TREATMENT, seed)
        try:
            # load the test prompt + extraction prompt
            prompt, extraction_prompt = (
                self.PROMPTS["decision_prompt"],
                self.PROMPTS["extraction_prompt"],
            )
            test_prompt = prompt.replace(
                "{{test_case}}",
                test_case.TREATMENT.format(
                    insert_headings=True, show_type=False, show_generated=True   # TODO show_generated should be False here as the LLM is not supposed to see which parts are generated and which aren't
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
                    None,
                    treatment_options,
                    treatment_answer,
                    treatment_decision,
                    None,
                    None,
                )
            else:
                test_prompt = prompt.replace(
                    "{{test_case}}",
                    test_case.CONTROL.format(
                        insert_headings=True, show_type=False, show_generated=True   # TODO show_generated should be False here as the LLM is not supposed to see which parts are generated and which aren't
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
                    control_answer,
                    control_decision,
                    treatment_options,
                    treatment_answer,
                    treatment_decision,
                    None,
                    None,
                )
        except Exception as e:
            raise DecisionError(e)
        
    def decide_all(self, test_cases: list[TestCase], seed: int = 42) -> list[DecisionResult]:
        """
        Function to decide on all test cases in the list.
        
        Args:
            test_cases (list[TestCase]): A list of test cases to decide on.
            seed (int): A seed for deterministic randomness
        """
        all_decisions = []
        for test_id, test_case in enumerate(test_cases):
            try:
                all_decisions.append(self.decide(test_case, seed))
            except DecisionError:
                print(f"Decision failed for the test case {test_id}")
                all_decisions.append(None)
            
        return all_decisions


class GptThreePointFiveTurbo(GPT):
    """
    A class representing a GPT-3.5-Turbo LLM that populates test cases according to the scenario 
    starting from the brackets that are either identical for both control and treatment or unique for control, 
    and then adding those unique for treatment.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(self):
        super().__init__()
        self.NAME = "gpt-3.5-turbo"


class GptFourO(GPT):
    """
    A class representing a GPT-4o LLM that populates test cases according to the scenario 
    starting from the brackets that are either identical for both control and treatment or unique for control, 
    and then adding those unique for treatment.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(self):
        super().__init__()
        self.NAME = "gpt-4o"
