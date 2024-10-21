from core.base import TestGenerator, LLM, RatioScaleMetric
import importlib
import os


# Define the template for the config.xml file
# This is used by the add_cognitive_bias function to add required files and folder structure for a new cognitive bias
config_file_template = """<config bias="{}">
    <custom_values name="my_custom_value">
        <value>first</value>
        <value>second</value>
    </custom_values>
    <variant name="default">
        <template type="control">
            <situation>Suppose you are a [[type]] manager at [[organization]].</situation>
            <prompt>Which option do you choose?</prompt>
            <option>First option</option>
            <option>Second option</option>
        </template>
        <template type="treatment">
            <situation>Suppose you are a [[type]] manager at [[organization]].</situation>
            <situation>You always choose the {} option.</situation>
            <prompt>Which option do you choose?</prompt>
            <option>First option</option>
            <option>Second option</option>
        </template>
    </variant>
</config>"""

# Define the template for the test.py file
# This is used by the add_cognitive_bias function to add required files and folder structure for a new cognitive bias
test_file_template = """from base import TestGenerator, LLM, Metric
from tests import TestCase, Template, TestConfig, DecisionResult
import numpy as np


class {}TestGenerator(TestGenerator):
    \"""
    Test generator for the {}.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for this cognitive bias.
    \"""

    def __init__(self):
        self.BIAS: str = "{}"
        self.config: TestConfig = super().load_config(self.BIAS)

    def generate_all(self, model: LLM, scenarios: list[str], seed: int = 42) -> list[TestCase]:
        # Load the custom values from the test config
        config_values = self.config.get_custom_values()   # TODO: Remove this line if custom values are not needed

        # Create test cases for all scenarios
        test_cases: list[TestCase] = []
        for scenario in scenarios:
            try:
                custom_values = {{
                    "custom_value": config_values["my_custom_value"][0]   # TODO: Remove this line if custom values are not needed
                }}

                test_case = self.generate(model, scenario, custom_values, seed)
                test_cases.append(test_case)
            except Exception as e:
                print(f"Generating the test case failed.\\nScenario: {{scenario}}\\nSeed: {{seed}}")
                print(e)

        return test_cases

    def generate(self, model: LLM, scenario: str, config_values: dict = {{}}, seed: int = 42) -> TestCase:
        # Load the control and treatment templates
        control: Template = self.config.get_control_template()       # TODO: Pass the variant name as a function parameter if you have more than one test variant
        treatment: Template = self.config.get_treatment_template()   # TODO: Pass the variant name as a function parameter if you have more than one test variant

        # Populate the templates with custom values
        treatment.insert_values([("my_custom_value", config_values["custom_value"])], kind='manual')   # TODO: Remove this line if custom values are not needed

        # Populate the templates using the model and the scenario
        control, treatment = super().populate(model, control, treatment, scenario)

        # Create a test case object
        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=treatment,
            generator=model.NAME,
            scenario=scenario,
            control_values=None,
            treatment_values=None,
            variant=None,
            remarks=None
        )

        return test_case


class {}Metric(Metric):

    def __init__(self):
        pass

    def _compute(self, test_result: tuple[TestCase, DecisionResult]) -> float:
        # Extract the test case and decision result from the tuple
        test_case: TestCase = test_result[0]
        decision_result: DecisionResult = test_result[1]

        # Calculate the biasedness
        biasedness = 0.0   # TODO: Implement calculation of biasedness here

        return biasedness

    def compute(self, test_results: list[tuple[TestCase, DecisionResult]]) -> float:
        # Calculate the average biasedness score across all tests
        biasedness_scores = [self._compute(test_result) for test_result in test_results]
        return np.mean(biasedness_scores)"""

def add_cognitive_bias(name: str) -> None:
    """
    Adds the folder and file structure for a new cognitive bias to this code base.

    Args:
        name (str): The name of the new cognitive bias.
    """

    # Transform the cognitive bias name into title case and determine the folder name
    name = name.title()
    name_no_spaces = name.replace(" ", "")
    folder_name = f"./tests/{name_no_spaces}"

    # Validate that the name only contains alphabetical characters
    if not name_no_spaces.isalpha():
        raise ValueError("Cognitive bias name must contain only alphabetical characters.")  
    
    # Validate that the folder does not yet exist
    if os.path.isdir(folder_name):
        raise FileExistsError(f"A folder {folder_name} already exists.")

    # Create a new folder for the cognitive bais
    os.makedirs(folder_name)

    # Create __init__.py file
    init_file_path = os.path.join(folder_name, "__init__.py")
    with open(init_file_path, 'w') as f:
        f.write("# __init__.py for the {}\n".format(name))

    # Create config.xml file
    config_file_path = os.path.join(folder_name, "config.xml")
    config_content = config_file_template.format(name, "{{my_custom_value}}")
    with open(config_file_path, 'w') as f:
        f.write(config_content)

    # Create test.py file
    test_file_path = os.path.join(folder_name, "test.py")
    test_content = test_file_template.format(name_no_spaces, name, name, name_no_spaces)
    with open(test_file_path, 'w') as f:
        f.write(test_content)

    print(f"Files for {name} successfully created under {folder_name}.")


def get_generator(bias: str) -> TestGenerator:
    """
    Returns a test generator for the specified cognitive bias.

    Args:
        bias (str): The name of the cognitive bias for which to get the test generator.

    Returns:
        A TestGenerator object for the specified cognitive bias.
    """
    try:
        # Construct the module path
        module_path = f'tests.{bias}.test'

        # Dynamically import the module
        module = importlib.import_module(module_path)

        # Construct the class name
        class_name = f'{bias}TestGenerator'

        # Get the class from the module
        TestGeneratorClass = getattr(module, class_name)

        return TestGeneratorClass()
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not find the generator for bias '{bias}': {e}")
    

def get_metric(bias: str) -> RatioScaleMetric:
    """
    Returns a metric for the specified cognitive bias.

    Args:
        bias (str): The name of the cognitive bias for which to get the metric generator.

    Returns:
        A Metric object for the specified cognitive bias.
    """
    try:
        # Construct the module path
        module_path = f'tests.{bias}.test'

        # Dynamically import the module
        module = importlib.import_module(module_path)

        # Construct the class name
        class_name = f'{bias}Metric'

        # Get the class from the module
        MetricClass = getattr(module, class_name)

        return MetricClass
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not find the metric for bias '{bias}': {e}")


def get_model(model_name: str, randomly_flip_options: bool = False, shuffle_answer_options: bool = False) -> LLM:
    """
    Returns a model instance of the specified type. Currently supported model names:
    - GPT-4o
    - GPT-4o-Mini
    - GPT-3.5-Turbo
    - Llama-3.1-8B
    - Llama-3.1-70B
    - Llama-3.1-405B
    - Llama-3.2-1B
    - Llama-3.2-3B
    - Llama-3.2-11B
    - Llama-3.2-90B
    - Gemini-1.5-Flash
    - Gemini-1.5-Flash-8B
    - Gemini-1.5-Pro
    - Claude-3.5-Sonnet
    - Claude-3.5-Haiku
    - Mistral-Large-2
    - Mistral-Small
    - Gemma-2-9B-IT
    - Gemma-2-27B-IT
    - Qwen-2.5-72B-Instruct
    - WizardLM-2-8x22B
    - WizardLM-2-7B
    - Phi-3-Vision-128K-Instruct
    - Yi-Large
    - Random

    Args:
        model_name (str): The name of the model.

    Returns:
        A LLM object for the specified model.
    """

    if model_name == "GPT-4o":
        from models.OpenAI.gpt import GptFourO
        return GptFourO(randomly_flip_options, shuffle_answer_options)
    elif model_name == "GPT-4o-Mini":
        from models.OpenAI.gpt import GptFourOMini
        return GptFourOMini(randomly_flip_options, shuffle_answer_options)
    elif model_name == "GPT-3.5-Turbo":
        from models.OpenAI.gpt import GptThreePointFiveTurbo
        return GptThreePointFiveTurbo(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Llama-3.1-8B":
        from models.Meta.model import LlamaThreePointOneEightB
        return LlamaThreePointOneEightB(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Llama-3.1-70B":
        from models.Meta.model import LlamaThreePointOneSeventyB
        return LlamaThreePointOneSeventyB(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Llama-3.1-405B":
        from models.Meta.model import LlamaThreePointOneFourHundredFiveB
        return LlamaThreePointOneFourHundredFiveB(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Llama-3.2-1B":
        from models.Meta.model import LlamaThreePointTwoOneB
        return LlamaThreePointTwoOneB(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Llama-3.2-3B":
        from models.Meta.model import LlamaThreePointTwoThreeB
        return LlamaThreePointTwoThreeB(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Llama-3.2-11B":
        from models.Meta.model import LlamaThreePointTwoElevenB
        return LlamaThreePointTwoElevenB(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Llama-3.2-90B":
        from models.Meta.model import LlamaThreePointTwoNinetyB
        return LlamaThreePointTwoNinetyB(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Gemini-1.5-Flash":
        from models.Google.model import GeminiOneFiveFlash
        return GeminiOneFiveFlash(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Gemini-1.5-Flash-8B":
        from models.Google.model import GeminiOneFiveFlashEightB
        return GeminiOneFiveFlashEightB(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Gemini-1.5-Pro":
        from models.Google.model import GeminiOneFivePro
        return GeminiOneFivePro(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Claude-3.5-Sonnet":
        from models.Anthropic.model import ClaudeThreeFiveSonnet
        return ClaudeThreeFiveSonnet(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Claude-3.5-Haiku":
        from models.Anthropic.model import ClaudeThreeHaiku
        return ClaudeThreeHaiku(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Mistral-Large-2":
        from models.MistralAI.model import MistralLargeTwo
        return MistralLargeTwo(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Mistral-Small":
        from models.MistralAI.model import MistralSmall
        return MistralSmall(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Gemma-2-9B-IT":
        from models.Google.model import GemmaTwoNineB
        return GemmaTwoNineB(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Gemma-2-27B-IT":
        from models.Google.model import GemmaTwoTwentySevenB
        return GemmaTwoTwentySevenB(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Qwen-2.5-72B-Instruct":
        from models.Alibaba.model import QwenTwoPointFiveSeventyTwoB
        return QwenTwoPointFiveSeventyTwoB(randomly_flip_options, shuffle_answer_options)
    elif model_name == "WizardLM-2-8x22B":
        from models.Microsoft.model import WizardLMTwoEightTwentyTwoB
        return WizardLMTwoEightTwentyTwoB(randomly_flip_options, shuffle_answer_options)
    elif model_name == "WizardLM-2-7B":
        from models.Microsoft.model import WizardLMTwoSevenB
        return WizardLMTwoSevenB(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Phi-3-Vision-128K-Instruct":
        from models.Microsoft.model import PhiThree
        return PhiThree(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Yi-Large":
        from models.ZeroOneAI.model import YiLarge
        return YiLarge(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Random":
        from models.Random.model import RandomModel
        return RandomModel(randomly_flip_options, shuffle_answer_options)
    
    raise ValueError(f"Model '{model_name}' is not supported.")


if __name__ == "__main__":
    print("You are running the script for adding a new cognitive bias test.")

    done = False
    while not done:
        try:
            bias_name = input("Enter the name of the cognitive bias: ")
            add_cognitive_bias(bias_name)
            done = True
        except (ValueError, FileExistsError) as e:
            print(e)