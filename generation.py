from abc import ABC, abstractmethod
from tests import TestCase, Template
from models import LLM


class TestGenerator(ABC):
    """
    Abstract base class for test generators. A test generator is responsible for generating test cases for a particular cognitive bias.
    
    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.    
    """

    def __init__(self):
        self.BIAS = "None"
    
    @abstractmethod
    def generate(self, model: LLM, bias_dict: dict, scenario: str) -> TestCase:
        """
        Generates a test case for the cognitive bias associated with this test generator.

        Args:
            model (LLM): The LLM model to use for generating the test case.
            bias_dict (dict): A dictionary containing the bias data for the test case from the respective YAML file.
            scenario (str): The scenario for which to generate the test case.

        Returns:
            A TestCase object representing the generated test case.
        """
        pass

    def populate(self, model: LLM, control: Template, treatment: Template, scenario: str) -> tuple[Template, Template]:
        """
        Populates the control and treatment templates using the provided LLM model and scenario.

        Args:
            model (LLM): The LLM model to use for populating the templates.
            control (Template): The control template.
            treatment (Template): The treatment template.
            scenario (str): The scenario for which to populate the templates.

        Returns:
            A tuple containing the populated control and treatment templates.
        """

        # Serialize the templates into strings
        control_str = control.serialize()
        treatment_str = treatment.serialize()

        # Populate the serialized templates using the model and scenario
        control_str, treatment_str = model.populate(control_str, treatment_str, scenario)

        # Deserialize the populated strings back into templates
        control, treatment = Template(control_str), Template(treatment_str)

        return control, treatment


class DummyBiasTestGenerator(TestGenerator):
    """
    Dummy test generator for generating test cases. This class is implemented for testing purposes only.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
    """

    def __init__(self):
        self.BIAS = "Dummy Bias"

    def generate(self, model: LLM, bias_dict: dict, scenario: str) -> TestCase:
        # Create a dummy template for the control variant of the test case
        control: Template = Template()
        control.add_situation(scenario)
        control.add_prompt(bias_dict['control']['prompt'][0])
        control.add_option(bias_dict['control']['option'][0])
        control.add_option(bias_dict['control']['option'][1])
        
        # Create a dummy template for the treatment variant of the test case
        treatment: Template = Template()
        treatment.add_situation(scenario)
        treatment.add_situation(bias_dict['treatment']['situation'][0])
        treatment.add_prompt(bias_dict['treatment']['prompt'][0])
        treatment.add_option(bias_dict['treatment']['option'][0])
        treatment.add_option(bias_dict['treatment']['option'][1])

        # Populate the templates using the model and scenario
        control, treatment = super().populate(model, control, treatment, scenario)

        # Create a test case object
        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=treatment,
            generator=model.NAME,
            scenario=scenario
        )

        return test_case
    

def get_generator(bias: str) -> TestGenerator:
    """
    Returns a test generator for the specified cognitive bias.

    Args:
        bias (str): The name of the cognitive bias for which to get the test generator.

    Returns:
        A TestGenerator object for the specified cognitive bias.
    """
    try:
        return globals()[f"{bias}TestGenerator"]()
    except KeyError:
        raise ValueError(f"Invalid (or not CamelCased) name of bias: {bias}")