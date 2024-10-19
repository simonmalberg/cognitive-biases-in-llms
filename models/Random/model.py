from core.base import LLM
import random
import yaml


class RandomModel(LLM):
    """
    A class representing a random model that populates test cases with random samples of words from the scenario. This class is implemented for testing purposes only.

    Attributes:
        NAME (str): The name of the model.
    """

    def __init__(self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options
        )
        self.NAME = "random-model"
        with open("./models/Random/prompts.yml") as f:
            self._PROMPTS = yaml.safe_load(f)

    def prompt(self, prompt: str, temperature: float = 0.0, seed: int = 42) -> str:
        """
        Generates a random response to the provided prompt.

        Args:
            prompt (str): The prompt to generate a response for.
            temperature (float): The temperature value, irrelevant.
            seed (int): The seed for controlling the LLM's output. It is used to perform random sampling.

        Returns:
            str: The response generated by the LLM.
        """
        seed = random.randint(0, 10000000000)
        random.seed(seed)
        # If there are 11 options, return the random integer from 0 to 10
        if "Option 11" in prompt: 
            return f"Option {str(random.randint(1, 11))}"
        # If there are only 7 options, return the random integer from 0 to 6
        else:
            return f"Option {str(random.randint(1, 7))}"
