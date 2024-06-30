from abc import ABC, abstractmethod
from tests import TestCase, Template, TestConfig
from models import LLM
import random
import re


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

    def populate(
        self, model: LLM, control: Template, treatment: Template, scenario: str
    ) -> tuple[Template, Template]:
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

        # Populate the templates using the model and scenario
        control, treatment, replacements = model.populate(control, treatment, scenario)

        return control, treatment, replacements


class DummyBiasTestGenerator(TestGenerator):
    """
    Dummy test generator for generating test cases. This class is implemented for testing purposes only.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Dummy Bias.
    """

    def __init__(self):
        self.BIAS = "Dummy Bias"
        self.config = super().load_config(self.BIAS)

    def generate(self, model: LLM, scenario: str) -> TestCase:
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
            control_custom_values=None,
            treatment_custom_values=None,
            scenario=scenario,
        )

        return test_case


# TODO: add functionality to generate the anchor from the answer options' interval
class AnchoringBiasTestGenerator(TestGenerator):
    """
    Test generator for the Anchoring Bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Anchoring Bias.
    """

    def __init__(self):
        self.BIAS = "Anchoring Bias"
        self.config = super().load_config(self.BIAS)

    def custom_population(self, model: LLM, completed_template: Template) -> None:
        """
        Custom population method for the Anchoring Bias test case.

        Args:
            model (LLM): The LLM model to use for generating the anchor sentence.
            completed_template (Template): The assembled template with scenario for the test case.
        """
        custom_values = self.config.get_custom_values()
        # Loading the anchor sentence generation prompt
        anchor_sentence = custom_values["anchor_sentence"][0]
        # generate the anchor sentence
        anchor_sentence = model.generate_misc(anchor_sentence)
        # Inserting the anchor into the template
        completed_template.insert_custom_values(["anchor_sentence"], [anchor_sentence])
        # Explicitly extract the numerical value from the generated anchor sentence
        anchor = re.findall(r"\d+", anchor_sentence)
        assert (
            len(anchor) == 1
        ), "The anchor sentence should contain exactly one numerical value"
        # technically, we don't insert anything (just remember the value in template)
        completed_template.insert_custom_values(["anchor"], anchor)

    def generate(self, model: LLM, scenario: str) -> TestCase:

        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # Insert custom anchor sentence and remember the anchor value
        self.custom_population(model, treatment)
        treatment_inserted_values = treatment.inserted_values

        control, treatment, replacements = super().populate(
            model, control, treatment, scenario
        )

        # Create a test case object
        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=treatment,
            generator=model.NAME,
            control_custom_values=None,
            treatment_custom_values=treatment_inserted_values,
            replacements=replacements,
            scenario=scenario,
        )

        return test_case


class LossAversionTestGenerator(TestGenerator):
    """
    Test generator for the Loss Aversion bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Loss Aversion bias.
    """

    def __init__(self):
        self.BIAS = "Loss Aversion"
        self.config = super().load_config(self.BIAS)

    def custom_population(self, completed_template: Template) -> None:
        """
        Custom population method for the Loss Aversion test case.

        Args:
            completed_template (Template): The assembled template for the test case.
        """
        # Loading the dict with custom values
        custom_values = self.config.get_custom_values()
        # Loading the possible outcomes and amounts
        outcome = custom_values["outcome"]
        amount = custom_values["amount"]

        # Sampling one of ['gain', 'loss'] and taking the index:
        first_outcome = random.choice(outcome)
        first_idx = outcome.index(first_outcome)
        # Taking the other outcome as the second one:
        second_outcome = outcome[(first_idx + 1) % 2]
        # Taking the respective amounts
        first_amount = amount[first_idx]
        second_amount = amount[(first_idx + 1) % 2]

        # Inserting the outcomes and amounts into the template
        patterns = ["first_outcome", "second_outcome", "first_amount", "second_amount"]
        values = [first_outcome, second_outcome, first_amount, second_amount]
        completed_template.insert_custom_values(patterns, values)

        # Sampling the value of lambda - TODO: might be better to sample a vector for several tests, discuss it
        lambda_coef = round(random.uniform(1, 2), 1)  # TODO: select the distribution
        completed_template.insert_custom_values(["lambda_coef"], [str(lambda_coef)])

    def generate(self, model: LLM, scenario: str) -> TestCase:

        treatment: Template = self.config.get_treatment_template()
        self.custom_population(treatment)
        treatment_custom_values = treatment.inserted_values

        _, treatment, replacements = super().populate(model, None, treatment, scenario)

        # Create a test case object and remember the sampled lambda value
        test_case = TestCase(
            bias=self.BIAS,
            control=None,
            treatment=treatment,
            generator=model.NAME,
            control_custom_values=None,
            treatment_custom_values=treatment_custom_values,
            replacements=replacements,
            scenario=scenario,
        )

        return test_case


class HaloEffectTestGenerator(TestGenerator):
    """
    Test generator for the Halo Effect bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Halo Effect bias.
    """

    def __init__(self):
        self.BIAS = "Halo Effect"
        self.config = super().load_config(self.BIAS)

    def custom_population(self, completed_template: Template) -> str:
        """
        Custom population method for the Halo Effect test case.

        Args:
            completed_template (Template): The assembled template for the test case.

        Returns:
            experience_sentiment (str): The sampled custom value used in both templates.
        """
        # Loading the dict with custom values
        custom_values = self.config.get_custom_values()
        experience_sentiments = custom_values["experience_sentiment"]
        # Sampling one of the outcomes
        experience_sentiment = random.choice(experience_sentiments)
        completed_template.insert_custom_values(
            ["experience_sentiment"], [experience_sentiment]
        )

        return experience_sentiment

    def generate(self, model: LLM, scenario: str) -> TestCase:

        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        # sample a sentiment for control version, insert it in the treatment
        experience_sentiment = self.custom_population(control)
        treatment.insert_custom_values(["experience_sentiment"], [experience_sentiment])
        # get dictionary of inserted values
        control_inserted_values = control.inserted_values
        treatment_inserted_values = treatment.inserted_values

        control, treatment, replacements = super().populate(
            model, control, treatment, scenario
        )

        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=treatment,
            generator=model.NAME,
            control_custom_values=control_inserted_values,
            treatment_custom_values=treatment_inserted_values,
            replacements=replacements,
            scenario=scenario,
        )

        return test_case


class ConfirmationBiasTestGenerator(TestGenerator):
    """
    Test generator for the Confirmation Bias.

    Attributes:
        BIAS (str): The cognitive bias associated with this test generator.
        config (TestConfig): The test configuration for the Confirmation Bias.
    """

    def __init__(self):
        self.BIAS = "Confirmation Bias"
        self.config = super().load_config(self.BIAS)

    def custom_population(self, completed_template: Template) -> None:
        """
        Custom population method for the Confirmation Bias test case.

        Args:
            completed_template (Template): The assembled template for the test case.
        """
        # Loading the dict with custom values
        custom_values = self.config.get_custom_values()
        # Loading the possible kinds of arguments
        outcomes = custom_values["argument"]
        num_arguments = len(outcomes)
        # Shuffling arguments
        random.shuffle(outcomes)
        # insertion of the arguments into the respective answer places of the template
        to_be_filled = [f"argument_{i}" for i in range(1, 2 * num_arguments + 1)]
        completed_template.insert_custom_values(to_be_filled, outcomes)

    def generate(self, model: LLM, scenario: str) -> TestCase:

        control: Template = self.config.get_control_template()
        treatment: Template = self.config.get_treatment_template()

        self.custom_population(treatment)

        treatment_inserted_values = treatment.inserted_values
        control, treatment, replacements = super().populate(
            model, control, treatment, scenario
        )

        test_case = TestCase(
            bias=self.BIAS,
            control=control,
            treatment=treatment,
            generator=model.NAME,
            control_custom_values=None,
            treatment_custom_values=treatment_inserted_values,
            replacements=replacements,
            scenario=scenario,
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
