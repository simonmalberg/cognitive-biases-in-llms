from mistralai import Mistral
from core.base import LLM
import yaml
import os


class MistralAI(LLM):
    """
    An abstract class representing MistralAI models provided by the MistralAI API.

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
        if "MISTRAL_API_KEY" not in os.environ:
            raise ValueError(
                "Cannot access MistralAI API due to missing API key. Please store your API key in environment variable 'MISTRAL_API_KEY'."
            )
        # The client object for the MistralAI API
        self._CLIENT = Mistral(
            api_key=os.environ["MISTRAL_API_KEY"],
        )
        self.RESPONSE_FORMAT = None
        with open("./models/MistralAI/prompts.yml") as f:
            self._PROMPTS = yaml.safe_load(f)

    def prompt(self, prompt: str, temperature: float = 0.0, seed: int = 42) -> str:
        """
        Function to prompt the model with a given prompt and return the response
        according to the original MistralAI inference pipeline.
        """
        chat_response = self._CLIENT.chat.complete(
            model=self.NAME,
            temperature=temperature,
            random_seed=seed,
            messages=[{"role": "user", "content": prompt}],
        )

        return chat_response.choices[0].message.content


class MistralLargeTwo(MistralAI):
    """
    A class representing a Mistral Large 2 LLM that decides on the test cases provided.

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
        self.NAME = "mistral-large-2407"


class MistralSmall(MistralAI):
    """
    A class representing a Mistral Small 24.09 LLM that decides on the test cases provided.

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
        self.NAME = "mistral-small-2409"
