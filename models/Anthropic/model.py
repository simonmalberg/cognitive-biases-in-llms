import anthropic
from tests import Template, TestCase, DecisionResult
from base import LLM, DecisionError
import yaml
import re


class Claude(LLM):
    """
    An abstract class representing a Claude model from Anthropic.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(self, shuffle_answer_options: bool = False):
        super().__init__(shuffle_answer_options=shuffle_answer_options)
        self._CLIENT = anthropic.Anthropic()
        with open("./models/Anthropic/prompts.yml") as f:
            self._PROMPTS = yaml.safe_load(f)

    def prompt(self, prompt: str, temperature: float = 0.7, seed: int = 42) -> str:
        """
        Function to prompt the model with a given prompt and return the response
        according to the Anthropic API pipeline.
        """
        # Anthropic API doesn't have a seed parameter
        message = self._CLIENT.messages.create(
            model=self.NAME,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        return message.content[0].text

    def decide_all(
        self, test_cases: list[TestCase], temperature: float = 0.7, seed: int = 42
    ) -> list[DecisionResult]:
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

    def decide(
        self, test_case: TestCase, temperature: float = 0.7, seed: int = 42
    ) -> DecisionResult:
        # Declare the results variables
        (
            control_answer,
            control_extraction,
            control_option,
            control_option_texts,
            control_option_order,
        ) = (None, None, None, [], [])
        (
            treatment_answer,
            treatment_extraction,
            treatment_option,
            treatment_option_texts,
            treatment_option_order,
        ) = (None, None, None, [], [])

        # Obtain decisions for the control and treatment decision-making tasks
        if test_case.CONTROL is not None:
            (
                control_answer,
                control_extraction,
                control_option,
                control_option_texts,
                control_option_order,
            ) = self._decide(test_case.CONTROL, temperature=temperature, seed=seed)
        if test_case.TREATMENT is not None:
            (
                treatment_answer,
                treatment_extraction,
                treatment_option,
                treatment_option_texts,
                treatment_option_order,
            ) = self._decide(test_case.TREATMENT, temperature=temperature, seed=seed)

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
            seed=seed,
        )

        return decision_result

    def populate(
        self,
        control: Template,
        treatment: Template,
        scenario: str,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> tuple[Template, Template]:
        return "Anthropic models are not used to populate test cases."

    def _decide(
        self, template: Template, temperature: float = 0.7, seed: int = 42
    ) -> tuple[str, str, int, list[str], list[int]]:
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
        decision_prompt = self._PROMPTS["decision_prompt"]
        extraction_prompt = self._PROMPTS["extraction_prompt"]

        # 2A. Format the template and insert it into the decision prompt
        decision_prompt = decision_prompt.replace(
            "{{test_case}}",
            template.format(shuffle_options=self.shuffle_answer_options, seed=seed),
        )
        options, option_order = template.get_options(
            shuffle_options=self.shuffle_answer_options, seed=seed
        )

        # 2B. Obtain a response from the LLM
        try:
            decision_response = self.prompt(
                decision_prompt, temperature=temperature, seed=seed
            )
        except Exception as e:
            raise DecisionError(
                f"Could not obtain a decision from the model to the following prompt:\n\n{decision_prompt}\n\n{e}"
            )

        # 3A. Insert the decision options and the decision response into the extraction prompt
        extraction_prompt = extraction_prompt.replace(
            "{{options}}",
            "\n".join(
                f"Option {index}: {option}"
                for index, option in enumerate(options, start=1)
            ),
        )
        extraction_prompt = extraction_prompt.replace("{{answer}}", decision_response)

        # 3B. Let the LLM extract the final chosen option from its previous answer
        try:
            extraction_response = self.prompt(
                extraction_prompt, temperature=temperature, seed=seed
            )
        except Exception as e:
            raise DecisionError(
                f"An error occurred while trying to extract the chosen option with the following prompt:\n\n{extraction_prompt}\n\n{e}"
            )

        # 3C. Extract the option number from the extraction response
        pattern = r"\b(?:[oO]ption) (\d+)\b"
        match = re.search(pattern, extraction_response)
        chosen_option = int(match.group(1)) if match else None

        if chosen_option is None:
            raise DecisionError(
                f"Could not extract the chosen option from the model's response:\n\n{decision_response}\n\n{extraction_response}\n\nNo option number detected in response."
            )

        return (
            decision_response,
            extraction_response,
            chosen_option,
            options,
            option_order,
        )


class ClaudeThreeFiveSonnet(Claude):
    """
    A class representing a Claude 3.5 Sonnet LLM that decides on the test cases provided.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(self, shuffle_answer_options: bool = False):
        super().__init__(shuffle_answer_options=shuffle_answer_options)
        self.NAME = "claude-3-5-sonnet-20240620"