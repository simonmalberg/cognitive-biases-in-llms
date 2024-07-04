import re
import xml.etree.ElementTree as ET


class DecisionResult:
    """
    A class representing the result of a decision made by an LLM for a specific test case.
    """

    def __init__(
        self,
        control_options: dict,
        control_decision: int,
        treatment_options: dict,
        treatment_decision: int,
        confidences: list,
        explanation: str,
    ):

        self.CONTROL_OPTIONS = control_options
        self.CONTROL_DECISION = control_decision
        self.TREATMENT_OPTIONS = treatment_options
        self.TREATMENT_DECISION = treatment_decision
        self.CONFIDENCES = None
        self.EXPLANATION = None

    def __str__(self) -> str:
        return f"---DecisionResult---\n\nCONTROL OPTIONS: {self.CONTROL_OPTIONS}\nCONTROL DECISION: {self.CONTROL_DECISION}\nTREATMENT OPTIONS: {self.TREATMENT_OPTIONS}\nTREATMENT DECISION: {self.TREATMENT_DECISION}\n\n------"

    def __repr__(self) -> str:
        return self.__str__()


class Template:  
    """
    A class representing a single template (e.g., the control or treatment variant) for a cognitive bias test case.
    """
    # TODO: Refactor to use ElementTree internally for easier handling, serialization, and parsing
    
    def __init__(self, from_string: str = None):
        if from_string is not None:
            self.elements = self.parse(from_string)
        else:
            self.elements = []
        # attribute for saving values generated manually/by LLM with a respective flag, e.g., key: [val, source]
        self.inserted_values = {}

    def add_situation(self, situation: str) -> None:
        self.elements.append((situation, 'situation'))
        self.validate(allow_incomplete=True)

    def add_prompt(self, prompt: str) -> None:
        self.elements.append((prompt, 'prompt'))
        self.validate(allow_incomplete=True)

    def add_option(self, option: str) -> None:
        self.elements.append((option, 'option'))
        self.validate(allow_incomplete=True)

    def format(self, insert_headings=True, show_type=False, show_generated=False) -> str:
        self.validate()

        # Function checks whether an element is the first of its type (e.g., 'situation', 'prompt', 'option') in the list
        def is_first_of_its_kind(element):
            for other in self.elements:
                if element == other:
                    return True
                elif element[1] == other[1]:
                    return False

        formatted = ''
        option_counter = 1

        # Iterate over all elements in this template and concatenate them to a string
        for element in self.elements:
            # If the element is the first of its type, insert a heading before it (if insert_headings=True)
            if insert_headings and is_first_of_its_kind(element):
                if formatted != '':
                    formatted += '\n'
                if element[1] == 'situation':
                    formatted += 'Situation:\n'
                elif element[1] == 'prompt':
                    formatted += 'Prompt:\n'
                elif element[1] == 'option':
                    formatted += 'Answer Options:\n'

            # If the element is an option, add a unique option number before it (if insert_headings=True)
            text = element[0]
            if insert_headings and element[1] == 'option':
                text = f'Option {option_counter}: {text}'
                option_counter += 1

            # Add HTML-like tags to the text based on the element's type (if show_type=True)
            if show_type:
                formatted += f'<{element[1]}>{text}</{element[1]}>\n'
            else:
                formatted += f'{text}\n'

        # Remove indicators for LLM-generated text (if show_generated=False)
        if not show_generated:
            formatted = formatted.replace('[[', '').replace(']]', '')

        return formatted

    def serialize(self) -> str:
        return self.format(insert_headings=False, show_type=True, show_generated=True)

    def parse(self, serialized_str: str) -> list[tuple[str, str]]:
        element_pattern = re.compile(r'<(situation|prompt|option)>(.*?)</\1>', re.DOTALL)
        elements = element_pattern.findall(serialized_str)
        return [(element[1], element[0]) for element in elements]

    def validate(self, allow_incomplete=False) -> bool:
        for element in self.elements:            
            # Validate that all elements are of type tuple[str, str]
            if not (isinstance(element, tuple) and len(element) == 2 and isinstance(element[0], str) and isinstance(element[1], str)):
                raise TypeError('All elements must be tuples of length 2, with a string as first and second element.')

            # Validate that all elements have a valid type
            if element[1] not in ['situation', 'prompt', 'option']:
                raise ValueError('Element type must be one of: situation, prompt, option.')

        # Validate that all element types appear in sufficient quantity
        if not allow_incomplete:
            if len([element for element in self.elements if element[1] == 'situation']) == 0:
                raise ValueError('At least one situation element must be provided.')
            if len([element for element in self.elements if element[1] == 'prompt']) == 0:
                raise ValueError('At least one prompt element must be provided.')
            if len([element for element in self.elements if element[1] == 'option']) < 2:
                raise ValueError('At least two option elements must be provided.')

        # Validate that option elements are never separated by other elements
        option_section_started = False
        for element in self.elements:
            if element[1] == 'option':
                if not option_section_started:
                    option_section_started = True
            else:
                if option_section_started:
                    raise ValueError('Option elements must not be separated by other elements.')

        return True
    
    def insert_values(self, pairs: list[tuple[str, str]], kind: str) -> None:
        """
        Inserts custom or generated values into the template.
        
        Args:
            pairs (list[tuple[str, str]]): A list of tuples, where each tuple contains a pattern and a value.
            kind (str): The kind of generation for the values being inserted: "manual" or "LLM".   
        """
        for pattern, value in pairs:
            for idx, _ in enumerate(self.elements):
                current = self.elements[idx][0]
                if pattern in current:
                    self.inserted_values[pattern] = (value, kind)
                    if kind == 'manual':
                        pattern = '{{' + pattern + '}}'
                        # for manual values, if value is None, replace with empty string (reason: Loss Aversion test)
                        value = '' if not value else value 
                    self.elements[idx] = (current.replace(pattern, value),) + self.elements[idx][1:]

    def __str__(self) -> str:
        return self.format(insert_headings=True, show_type=False, show_generated=False)

    def __repr__(self) -> str:
        return self.format(insert_headings=False, show_type=True, show_generated=True)


class TestConfig:
    """
    A class representing a configuration file for a cognitive bias test.

    Attributes:
        path (str): The path to the XML configuration file.
        config (ET): An ElementTree object representing the XML configuration file.
    """

    def __init__(self, path: str):
        self.path = path
        self.config = self.load(self.path)
    
    def load(self, path: str) -> ET:
        """
        Loads the XML configuration file for the specified cognitive bias.

        Args:
            bias (str): The name of the cognitive bias for which to load the configuration.

        Returns:
            An ElementTree object representing the XML configuration file.
        """
        return ET.parse(path)

    def get_bias_name(self) -> str:
        """
        Returns the name of the cognitive bias being tested.

        Returns:
            The name of the cognitive bias being tested.
        """
        return self.config.getroot().get('bias')
    
    def get_custom_values(self) -> dict:
        """
        Returns the custom values defined in the configuration file.

        Returns:
            A dictionary containing the custom values defined in the configuration file.
        """
        custom_values = self.config.getroot().findall('custom_values')
        custom_values_dict = {}
        for custom_value in custom_values:
            key = custom_value.get('name')
            custom_values_dict[key] = []
            for value in custom_value:
                custom_values_dict[key].append(value.text)
        
        return custom_values_dict

    def get_template(self, template_type: str = "control", variant: str = None) -> Template:
        """
        Returns a template from the test configuration.

        Args:
            template_type (str): The type of the template, e.g., "control" or "treatment".
            variant (str): The name of the variant. Defaults to None. Only needed if the test configuration includes multiple variants.

        Returns:
            A Template object representing the control template.
        """
        root = self.config.getroot()
        variants = list(root.findall('variant'))

        if not variants:
            # No variant elements, treat the root as the variant element
            variant_element = root
        elif len(variants) == 1:
            # Only one variant element present
            variant_element = variants[0]
        else:
            # Multiple variant elements, find the specified one
            variant_element = None
            for v in variants:
                if v.get('name') == variant:
                    variant_element = v
                    break
            if variant_element is None:
                raise ValueError(f"Variant '{variant}' not found in the configuration.")

        # Find the template with the specified type
        template_config = variant_element.find(f"template[@type='{template_type}']")
        if template_config is None:
            raise ValueError(f"No template with type '{template_type}' found in variant '{variant}'.")

        # Extract the components of the template
        template = Template()
        for c in list(template_config):
            if c.tag == 'situation':
                template.add_situation(c.text)
            elif c.tag == 'prompt':
                template.add_prompt(c.text)
            elif c.tag == 'option':
                template.add_option(c.text)

        return template

    def get_control_template(self, variant: str = None) -> Template:
        """
        Returns the control template for the specified variant or the Default variant if none is specified.

        Args:
            variant (str): The name of the variant. Defaults to "Default".

        Returns:
            A Template object representing the control template.
        """
        return self.get_template("control", variant)

    def get_treatment_template(self, variant: str = None) -> Template:
        """
        Returns the treatment template for the specified variant or the Default variant if none is specified.

        Args:
            variant (str): The name of the variant. Defaults to "Default".

        Returns:
            A Template object representing the treatment template.
        """
        return self.get_template("treatment", variant)


class TestCase:
    """
    A class representing a cognitive bias test case.

    Attributes:
        BIAS (str): The name of the cognitive bias being tested.
        CONTROL (Template): The control template for the test case.
        TREATMENT (Template): The treatment template for the test case.
        GENERATOR (str): The name of the LLM generator used to generate the treatment template.
        SCENARIO (str): The scenario in which the test case is being conducted.
        CONTROL_CUSTOM_VALUES (dict, optional): Values inserted used in the control template of the test case.
        TREATMENT_CUSTOM_VALUES (dict, optional): Values inserted in the treatment template of the test case.
        VARIANT (str, optional): The variant of the test case.
        REMARKS (str, optional): Any additional remarks about the test case.
    """

    def __init__(self, bias: str, control: Template, treatment: Template, generator: str, 
                 scenario: str, control_values: dict = None, treatment_values: dict = None,
                 variant: str = None, remarks: str = None):
        self.BIAS: str = bias
        self.CONTROL: Template = control
        self.TREATMENT: Template = treatment
        self.GENERATOR: str = generator
        self.SCENARIO: str = scenario
        self.CONTROL_VALUES: dict = control_values
        self.TREATMENT_VALUES: dict = treatment_values
        self.VARIANT: str = variant
        self.REMARKS: str = remarks

    def __str__(self) -> str:
        return f'---TestCase---\n\nBIAS: {self.BIAS}\nVARIANT: {self.VARIANT}\nSCENARIO: {self.SCENARIO}\nGENERATOR: {self.GENERATOR}\nCONTROL_VALUES: {self.CONTROL_VALUES}\nTREATMENT_VALUES: {self.TREATMENT_VALUES}\n\nCONTROL:\n{self.CONTROL}\n\nTREATMENT:\n{self.TREATMENT}\n\nREMARKS:\n{self.REMARKS}\n\n------'

    def __repr__(self) -> str:
        return self.__str__()
