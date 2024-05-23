from abc import ABC, abstractmethod
from tests import TestCase, Template, TemplateNew, TestConfig
from models import LLM
import random
import xml.etree.ElementTree as ET


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

    def load_config(self, bias: str) -> TestConfig:
        """
        Loads the test configuration from the specified XML file.

        Args:
            path (str): The path to the XML file containing the test configuration.

        Returns:
            A TestConfig object representing the loaded test configuration.
        """
        return TestConfig(f"./biases/{bias.replace(' ', '')}.xml")

    # TODO: irrelevant here, as the scenarios are inserted in the TemplateNew class
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
        self.config = super().load_config(self.BIAS)

    def generate(self, model: LLM, bias_dict: dict, scenario: str) -> TestCase:
        # Load the control and treatment templates from the test configuration
        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

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
    

class AnchoringBiasTestGenerator(TestGenerator):
    """
    Test generator for the Anchoring bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
    """

    def __init__(self):
        self.BIAS = "Anchoring Bias"

    def custom_population(self, model: LLM, bias_dict: dict, completed_template: str) -> str:
        """
        Custom population method for the Anchoring Bias test case.

        Args:
            bias_dict (dict): A dictionary containing the data for the test case from the respective YAML file.
            completed_template (str): The assembled template with scenario for the test case.

        Returns:
            The template populated with generated anchor sentence for the Anchoring Bias test case.
        """
        # Loading the anchor sentence generation prompt
        anchor_sentence = bias_dict['custom_values']['anchor_sentence'][0]
        # Generation of the anchor sentence
        anchor_sentence = model.populate(anchor_sentence, '', '')[0]

        # Inserting the anchor into the template
        completed_template = completed_template.replace('{anchor_sentence}', anchor_sentence)

        # TODO: discuss ways to track the inserted numerical value (e.g., in the remarks field)

        return completed_template

    def generate(self, model: LLM, bias_dict: dict, scenario: str) -> TestCase:
        # Create a template for both variants of the test case, filling in the scenario
        template = TemplateNew(bias_dict['control'], scenario)
        control = template.complete_template()

        template = TemplateNew(bias_dict['treatment'], scenario)
        treatment = template.complete_template()

        # Insert custom anchor sentence
        treatment = self.custom_population(model, bias_dict, treatment)

        # TODO: Populate the completed templates using LLM
        # control, treatment = model.populate(control, treatment, None)
        

        # Create a test case object
        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=treatment,
            generator=model.NAME,
            scenario=scenario
        )

        return test_case
    

class LossAversionTestGenerator(TestGenerator):
    """
    Test generator for the Loss Aversion bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
    """

    def __init__(self):
        self.BIAS = "Loss Aversion"

    def custom_population(self, bias_dict: dict, completed_template: str) -> str:
        """
        Custom population method for the Loss Aversion test case.

        Args:
            bias_dict (dict): A dictionary containing the data for the test case from the respective YAML file.
            completed_template (str): The assembled template with scenario for the test case.

        Returns:
            The template populated with custom values for the Loss Aversion test case.
        """
        # Loading the possible outcomes
        outcomes = bias_dict['custom_values']['outcome']
        amount = bias_dict['custom_values']['amount']

        # Sampling one of ['gain', 'loss'] and taking the index:
        first_outcome = random.choice(outcomes)
        first_idx = outcomes.index(first_outcome)
        # Taking the other outcome as the second one:
        second_outcome = outcomes[(first_idx + 1) % 2]
        # Taking the respective amounts
        first_amount = amount[first_idx]
        second_amount = amount[(first_idx + 1) % 2]

        # Inserting the outcomes and amounts into the template
        completed_template = completed_template.replace('{first_outcome}', first_outcome)
        completed_template = completed_template.replace('{second_outcome}', second_outcome)
        completed_template = completed_template.replace('{first_amount}', first_amount)
        completed_template = completed_template.replace('{second_amount}', second_amount)

        # Sampling the value of lambda - TODO: might be better to sample a vector for several
        # tests, discuss it. Besides, TODO: discuss ways to track the inserted values (e.g., in the remarks field)
        lambda_coef = round(random.uniform(1, 2), 1) # TODO: select the distribution
        completed_template = completed_template.replace('lambda_coef', str(lambda_coef))

        return completed_template
        

    def generate(self, model: LLM, bias_dict: dict, scenario: str) -> TestCase:
        # Create a template for the only variant of the test case, filling in the scenario
        template = TemplateNew(bias_dict['control'], scenario)
        control = template.complete_template()

        # Insert custom values for the Loss Aversion test
        control = self.custom_population(bias_dict, control)
        # TODO: Populate the completed template using LLM
        # control = model.populate(control, None, None)

        # Create a test case object and remember the sampled lambda value
        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=None,
            generator=model.NAME,
            scenario=scenario
        )

        return test_case
    

class HaloEffectTestGenerator(TestGenerator):
    """
    Test generator for the Halo Effect bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
    """

    def __init__(self):
        self.BIAS = "Halo Effect"

    def custom_population(self, bias_dict: dict, completed_template: str) -> str:
        """
        Custom population method for the Halo Effect test case.

        Args:
            bias_dict (dict): A dictionary containing the data for the test case from the respective YAML file.
            completed_template (str): The assembled template with scenario for the test case.

        Returns:
            The template populated with custom sentiment for the Halo Effect test case.
        """
        # Loading the possible outcomes
        experience_sentiments = bias_dict['custom_values']['experience_sentiment']
        # Sampling one of the outcomes
        experience_sentiment = random.choice(experience_sentiments)

        # Inserting the outcome into the template
        completed_template = completed_template.replace('{experience_sentiment}', experience_sentiment)

        return completed_template

    def generate(self, model: LLM, bias_dict: dict, scenario: str) -> TestCase:
        
        template = TemplateNew(bias_dict['control'], scenario)
        control = template.complete_template()

        template = TemplateNew(bias_dict['treatment'], scenario)
        treatment = template.complete_template()

        control = self.custom_population(bias_dict, control)
        treatment = self.custom_population(bias_dict, treatment)

        # TODO: Populate the completed template using LLM
        # control, treatment = model.populate(control, treatment, None)

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
