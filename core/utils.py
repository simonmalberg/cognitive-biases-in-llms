from core.base import TestGenerator, LLM, RatioScaleMetric
import importlib


# A list of all currently supported models
SUPPORTED_MODELS = [
    "GPT-4o",
    "GPT-4o-Mini",
    "GPT-3.5-Turbo",
    "Llama-3.1-8B",
    "Llama-3.1-70B",
    "Llama-3.1-405B",
    "Llama-3.2-1B",
    "Llama-3.2-3B",
    "Llama-3.2-11B",
    "Llama-3.2-90B",
    "Gemini-1.5-Flash",
    "Gemini-1.5-Flash-8B",
    "Gemini-1.5-Pro",
    "Claude-3.5-Sonnet",
    "Claude-3.5-Haiku",
    "Mistral-Large-2",
    "Mistral-Small",
    "Gemma-2-9B-IT",
    "Gemma-2-27B-IT",
    "Qwen-2.5-72B-Instruct",
    "WizardLM-2-8x22B",
    "WizardLM-2-7B",
    "Phi-3-Vision-128K-Instruct",
    "Yi-Large",
    "Random"
]


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
    Returns a model instance of the specified type. See utils.SUPPORTED_MODELS for a list of all currently supported model.

    Args:
        model_name (str): The name of the model. One from utils.SUPPORTED_MODELS.

    Returns:
        A LLM object for the specified model.
    """

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_name}' is not supported. Please choose one of: {SUPPORTED_MODELS}")
    
    if model_name == "GPT-4o":
        from models.OpenAI.model import GptFourO
        return GptFourO(randomly_flip_options, shuffle_answer_options)
    elif model_name == "GPT-4o-Mini":
        from models.OpenAI.model import GptFourOMini
        return GptFourOMini(randomly_flip_options, shuffle_answer_options)
    elif model_name == "GPT-3.5-Turbo":
        from models.OpenAI.model import GptThreePointFiveTurbo
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


def get_supported_models() -> list[str]:
    """
    Returns a list of all currently supported models.

    Returns:
        list[str]: A list with names of all currently supported models.
    """

    return SUPPORTED_MODELS