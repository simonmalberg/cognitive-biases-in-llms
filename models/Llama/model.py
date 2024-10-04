from tests import Template, TestCase, DecisionResult
from base import LLM, PopulationError, DecisionError
import re
import random
from openai import OpenAI
import json
import yaml
import warnings
import os


class FireworksAI(LLM):
    """
    An abstract class representing an LLM accessed through the fireworks.ai API.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False):
        super().__init__(randomly_flip_options=randomly_flip_options, shuffle_answer_options=shuffle_answer_options)

        if "FIREWORKS_AI_API_KEY" not in os.environ:
            raise ValueError("Cannot access Fireworks.ai API due to missing API key. Please store your API key in environment variable 'FIREWORKS_AI_API_KEY'.")

        self._CLIENT = OpenAI(
            base_url = "https://api.fireworks.ai/inference/v1",
            api_key = os.environ["FIREWORKS_AI_API_KEY"],
        )
        self.RESPONSE_FORMAT = None
        with open("./models/OpenAI/prompts.yml") as f:
            self._PROMPTS = yaml.safe_load(f)

    def prompt(self, prompt: str, temperature: float = 0.7, seed: int = 42) -> str:
        # Call the chat completions API endpoint
        response = self._CLIENT.chat.completions.create(
            model=self.NAME,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract and return the answer
        return response.choices[0].message.content

    def populate(self, control: Template, treatment: Template, scenario: str, temperature: float = 0.7, seed: int = 42) -> tuple[Template, Template]:
        # 1. Populate the gaps in the control template based on the scenario
        if control is not None and len(control.get_gaps()) > 0:
            control = self._populate(control, scenario, temperature=temperature, seed=seed)

        # 2. Fill the gaps in the treatment template that are shared with the control template
        if control is not None and treatment is not None:
            insertions = control.get_insertions(origin='model')
            treatment.insert(insertions=insertions)

        # 3. Populate the remaining gaps in the treatment template based on the scenario
        if treatment is not None and len(treatment.get_gaps()) > 0:
            treatment = self._populate(treatment, scenario, temperature=temperature, seed=seed)

        return control, treatment
    
    def decide_all(self, test_cases: list[TestCase], temperature: float = 0.7, seed: int = 42) -> list[DecisionResult]:
        """
        Function to decide on all test cases in the list.
        
        Args:
            test_cases (list[TestCase]): A list of test cases to decide on.
            seed (int): A seed for deterministic randomness
        """
        all_decisions = []
        for test_id, test_case in enumerate(test_cases):
            try:
                all_decisions.append(self.decide(test_case, temperature, seed))
            except DecisionError as e:
                print(f"Decision failed for the test case {test_id}. Error: {e}")
                all_decisions.append(None)
            
        return all_decisions

    def decide(self, test_case: TestCase, temperature: float = 0.7, seed: int = 42) -> DecisionResult:
        # Declare the results variables
        control_answer, control_extraction, control_option, control_option_texts, control_option_order = None, None, None, [], []
        treatment_answer, treatment_extraction, treatment_option, treatment_option_texts, treatment_option_order = None, None, None, [], []

        # Obtain decisions for the control and treatment decision-making tasks
        if test_case.CONTROL is not None:
            control_answer, control_extraction, control_option, control_option_texts, control_option_order = self._decide(test_case.CONTROL, temperature=temperature, seed=seed)
        if test_case.TREATMENT is not None:
            treatment_answer, treatment_extraction, treatment_option, treatment_option_texts, treatment_option_order = self._decide(test_case.TREATMENT, temperature=temperature, seed=seed)
        
        # Create a DecisionResult object containing the final decisions
        decision_result = DecisionResult(
            model=self.NAME,
            control_options=control_option_texts,
            control_option_order=control_option_order,
            control_answer=control_answer,
            control_decision=control_option,
            treatment_options=treatment_option_texts,
            treatment_option_order=treatment_option_order,
            treatment_answer=treatment_answer,
            treatment_decision=treatment_option,
            temperature=temperature,
            seed=seed
        )

        return decision_result

    def _define_response_format(self, mode: str, gaps: list[str]) -> object:
        """
        Defines the response format for controlled model outputs (special feature supported by OpenAI models).
        See https://openai.com/index/introducing-structured-outputs-in-the-api/

        Args:
            mode (str): The mode for controlling outputs, either "json_schema" (for Structured Outputs) or "json_object" (for JSON Mode).
            gaps (list[str]): A list of gaps in the template that need to be filled.

        Returns:
            object: An object describing the response format.
        """
        
        # Compile the response format for controlled model outputs (special feature supported by OpenAI models)
        if mode == "json_schema":
            # Works with gpt-4o-2024-08-06 and gpt-4o-mini-2024-07-18, see https://openai.com/index/introducing-structured-outputs-in-the-api/
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "population_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {gap: {"type": "string"} for gap in gaps},
                        "required": gaps,
                        "additionalProperties": False
                    }
                }
            }
        elif mode == "json_object":
            # Enable OpenAI's JSON model for models that do support it but don't support structured output (i.e., json_schema)
            response_format = {
                "type": "json_object"
            }
        else:
            response_format = None

        return response_format

    def _populate(self, template: Template, scenario: str, temperature: float = 0.7, seed: int = 42) -> Template:
        """
        Populates the blanks in the provided template according to the scenario.
        
        Args:
            template (Template): The template to populate.
            scenario (str): A string describing the scenario/context for the population.
            temperature (float): The temperature value of the LLM.
            seed (int): The seed for controlling the LLM's output.

        Returns:
            Template: The populated Template object.
        """

        # Load the system and user prompts
        system_prompt = self._PROMPTS['system_prompt']
        user_prompt = self._PROMPTS['population_prompt']

        # Compile the format instructions (JSON format) based on the remaining gaps in the template
        gaps = template.get_gaps(origin='model')
        gaps_format = [f"    \"{gap}\": \"...\"" for gap in gaps]
        expected_format = "{\n" + ',\n'.join(gaps_format) + "\n}"

        # Compile the response format for controlled model outputs (special feature supported by OpenAI models)
        response_format = self._define_response_format(self.RESPONSE_FORMAT, gaps)

        # Insert the scenario, template, and format instructions into the prompt
        user_prompt = user_prompt.replace("{{scenario}}", scenario)
        user_prompt = user_prompt.replace("{{template}}", template.format())
        user_prompt = user_prompt.replace("{{format}}", expected_format)

        # Obtain a response from the LLM
        response = self._CLIENT.chat.completions.create(
            model=self.NAME,
            temperature=temperature,
            response_format=response_format,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        # Parse the insertions generated by the LLM and fix them, if needed (adding missing double square brackets, which the LLM sometimes forgets)
        insertions = json.loads(response.choices[0].message.content)
        insertions = self._fix_insertions(insertions)

        # Validate the insertions
        self._validate_population(template, insertions, response.choices[0].message.content)

        # Make the insertions in the template
        for pattern in insertions.keys():
            template.insert(pattern, insertions[pattern], 'model')

        return template

    def _decide(self, template: Template, temperature: float = 0.7, seed: int = 42) -> tuple[str, str, int, list[str], list[int]]:
        """
        Prompts the model to choose one answer option from a decision-making task defined in the provided template.

        The decision is obtained through a two-step prompt: First, the model is presented with the decision-making test and can respond freely. Second, the model is instructed to extract the final answer from its previous response.

        Args:
            template (Template): The template defining the decision-making task.
            temperature (float): The temperature value of the LLM.
            seed (int): The seed for controlling the LLM's output.

        Returns:
            tuple[str, str, int, list[str], list[int]]: The raw model response, the model's extraction response, the number of the selected option (None if no selected option could be extracted), the answer option texts, and the order of answer options.
        """

        # 1. Load the decision and extraction prompts
        decision_prompt = self._PROMPTS['decision_prompt']
        extraction_prompt = self._PROMPTS['extraction_prompt']

        # 2A. Format the template and insert it into the decision prompt
        decision_prompt = decision_prompt.replace("{{test_case}}", template.format(randomly_flip_options=self.randomly_flip_options, shuffle_options=self.shuffle_answer_options, seed=seed))
        options, option_order = template.get_options(randomly_flip_options=self.randomly_flip_options, shuffle_options=self.shuffle_answer_options, seed=seed)

        # 2B. Obtain a response from the LLM
        try:
            decision_response = self.prompt(decision_prompt, temperature=temperature, seed=seed)
        except Exception as e:
            raise DecisionError(f"Could not obtain a decision from the model to the following prompt:\n\n{decision_prompt}\n\n{e}")

        # 3A. Insert the decision options and the decision response into the extraction prompt
        extraction_prompt = extraction_prompt.replace("{{options}}", "\n".join(f"Option {index}: {option}" for index, option in enumerate(options, start=1)))
        extraction_prompt = extraction_prompt.replace("{{answer}}", decision_response)

        # 3B. Let the LLM extract the final chosen option from its previous answer
        try:
            extraction_response = self.prompt(extraction_prompt, temperature=temperature, seed=seed)
        except Exception as e:
            raise DecisionError(f"An error occurred while trying to extract the chosen option with the following prompt:\n\n{extraction_prompt}\n\n{e}")

        # 3C. Extract the option number from the extraction response
        pattern = r'\b(?:[oO]ption) (\d+)\b'
        match = re.search(pattern, extraction_response)
        chosen_option = int(match.group(1)) if match else None

        if chosen_option is None:
            raise DecisionError(f"Could not extract the chosen option from the model's response:\n\n{decision_response}\n\n{extraction_response}\n\nNo option number detected in response.")

        return decision_response, extraction_response, chosen_option, options, option_order

    def _fix_insertions(self, insertions: dict) -> dict:
        """
        Ensures all keys in the dictionary start with [[ and end with ]].
        
        Args:
            insertions (dict): The dictionary to format keys for.

        Returns:
            dict: A new dictionary with formatted keys.
        """

        fixed_insertions = {}

        for key, value in insertions.items():
            # Check if the key already starts with [[ and ends with ]]
            if not key.startswith('[['):
                key = '[[' + key
            if not key.endswith(']]'):
                key = key + ']]'
            
            # Add the formatted key to the new dictionary
            fixed_insertions[key] = value
        
        return fixed_insertions


class LlamaThreePointOneEightB(FireworksAI):
    """
    A class representing a Llama-3.1-8B LLM.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False):
        super().__init__(randomly_flip_options=randomly_flip_options, shuffle_answer_options=shuffle_answer_options)
        self.NAME = "accounts/fireworks/models/llama-v3p1-8b-instruct"
        self.RESPONSE_FORMAT = None


class LlamaThreePointOneSeventyB(FireworksAI):
    """
    A class representing a Llama-3.1-70B LLM.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False):
        super().__init__(randomly_flip_options=randomly_flip_options, shuffle_answer_options=shuffle_answer_options)
        self.NAME = "accounts/fireworks/models/llama-v3p1-70b-instruct"
        self.RESPONSE_FORMAT = None


class LlamaThreePointOneFourHundredFiveB(FireworksAI):
    """
    A class representing a Llama-3.1-8B LLM.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False):
        super().__init__(randomly_flip_options=randomly_flip_options, shuffle_answer_options=shuffle_answer_options)
        self.NAME = "accounts/fireworks/models/llama-v3p1-405b-instruct"
        self.RESPONSE_FORMAT = None