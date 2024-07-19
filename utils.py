from base import TestGenerator, Metric
import importlib


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
        module_path = f'biases.{bias}.test'

        # Dynamically import the module
        module = importlib.import_module(module_path)

        # Construct the class name
        class_name = f'{bias}TestGenerator'

        # Get the class from the module
        TestGeneratorClass = getattr(module, class_name)

        return TestGeneratorClass()
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not find the generator for bias '{bias}': {e}")
    
def get_metric(bias: str) -> Metric:
    """
    Returns a metric for the specified cognitive bias.
    Args:
        bias (str): The name of the cognitive bias for which to get the metric generator.
    Returns:
        A Metric object for the specified cognitive bias.
    """
    try:
        # Construct the module path
        module_path = f'biases.{bias}.test'

        # Dynamically import the module
        module = importlib.import_module(module_path)

        # Construct the class name
        class_name = f'{bias}Metric'

        # Get the class from the module
        MetricClass = getattr(module, class_name)

        return MetricClass()
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not find the metric for bias '{bias}': {e}")