from base import LLM
from openai import OpenAI
import yaml
import os


class WizardLM(LLM):
    """
    An abstract class representing WizardLM models by Microsoft provided by the DeepInfra API.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        if "DEEPINFRA_API" not in os.environ:
            raise ValueError(
                "Cannot access DeepInfra API due to missing API key. Please store your API key in environment variable 'DEEPINFRA_API'."
            )
        # The client object for the DeepInfra API (supports OpenAI API)
        self._CLIENT = OpenAI(
            base_url="https://api.deepinfra.com/v1/openai",
            api_key=os.environ["DEEPINFRA_API"],
        )
        self.RESPONSE_FORMAT = None
        with open("./models/Microsoft/prompts.yml") as f:
            self._PROMPTS = yaml.safe_load(f)

    def prompt(self, prompt: str, temperature: float = 0.0, seed: int = 42) -> str:
        """
        Generates a response to the provided prompt.

        Args:
            prompt (str): The prompt to generate a response for.
            temperature (float): The temperature value of the LLM. For strictly decision models, we use a temperature of 0.0.
            seed (int): The seed for controlling the LLM's output. It is not used in WizardLM models.

        Returns:
            str: The response generated by the LLM.
        """
        # Call the chat completions API endpoint
        response = self._CLIENT.chat.completions.create(
            model=self.NAME,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract and return the answer
        return response.choices[0].message.content


class WizardLMTwoEightTwentyTwoB(WizardLM):
    """
    A class representing a WizardLM-2-8x22B LLM.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "microsoft/WizardLM-2-8x22B"


class WizardLMTwoSevenB(WizardLM):
    """
    A class representing a WizardLM-2-7B LLM.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        self.NAME = "microsoft/WizardLM-2-7B"