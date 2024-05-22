import re


class Template:  
    """
    A class representing a single template (e.g., the control or treatment variant) for a cognitive bias test case.
    """
    
    def __init__(self, from_string: str = None):
        if from_string is not None:
            self.elements = self.parse(from_string)
        else:
            self.elements = []

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

    def __str__(self) -> str:
        return self.format(insert_headings=True, show_type=False, show_generated=False)

    def __repr__(self) -> str:
        return self.format(insert_headings=False, show_type=True, show_generated=True)


class TemplateNew:
    """
    A (new) class representing a single template (e.g., the control or treatment variant) for a cognitive bias test case.

    Attributes:
        allowed_parts (list[str]): A list of strings specifying the allowed types of elements in the template.
        variant (dict): A dictionary representing the template variant data from YAML.
        scenario (str): The scenario for the given test template
    """

    def __init__(self, variant: dict, scenario: str, allowed_parts: list[str] = ['Situation', 'Question', 'Answer options']):
        self.allowed_parts = allowed_parts
        self.variant = variant
        self.scenario = scenario.lower()

    def complete_template(self) -> str:
        prev_type = self.variant[0]['type']
        assert prev_type == 'Generation', f"First block type must be 'Generation', not {prev_type}" 
        assembled_text = self.variant[0]['content']
        for block in self.variant[1:]:
            if block['type'] not in self.allowed_parts:
                print(f"Block type {block['type']} not in allowed parts {self.allowed_parts}, skipping it")
            else:
                if block['type'] == prev_type:
                    assembled_text = ' '.join((assembled_text, block['content']))
                else:
                    assembled_text = '\n'.join((assembled_text, block['type'], block['content']))
                prev_type = block['type']
        
        # insert scenario
        assembled_text = assembled_text.replace('{scenario}', self.scenario)

        return assembled_text


class TestCase:
    """
    A class representing a cognitive bias test case.

    Attributes:
        BIAS (str): The name of the cognitive bias being tested.
        CONTROL (Template): The control template for the test case.
        TREATMENT (Template): The treatment template for the test case.
        GENERATOR (str): The name of the LLM generator used to generate the treatment template.
        SCENARIO (str): The scenario in which the test case is being conducted.
        VARIANT (str, optional): The variant of the test case.
        REMARKS (str, optional): Any additional remarks about the test case.
    """

    def __init__(self, bias: str, control: Template, treatment: Template, generator: str, scenario: str, variant: str = None, remarks: str = None):
        self.BIAS: str = bias
        self.CONTROL: Template = control
        self.TREATMENT: Template = treatment
        self.GENERATOR: str = generator
        self.SCENARIO: str = scenario
        self.VARIANT: str = variant
        self.REMARKS: str = remarks

    def __str__(self) -> str:
        return f'---TestCase---\n\nBIAS: {self.BIAS}\nVARIANT: {self.VARIANT}\nSCENARIO: {self.SCENARIO}\nGENERATOR: {self.GENERATOR}\n\nCONTROL:\n{self.CONTROL}\n\nTREATMENT:\n{self.TREATMENT}\n\nREMARKS:\n{self.REMARKS}\n\n------'

    def __repr__(self) -> str:
        return self.__str__()
