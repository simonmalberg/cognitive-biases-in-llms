import xml.etree.ElementTree as ET
import warnings
import datetime
import random
import re


class DecisionResult:
    """
    A class representing the result of a decision made by an LLM for a specific test case.

    Attributes:
        MODEL (str): The name of the LLM used to make the decision.
        TEMPERATURE (float): The LLM temperature parameter used to generate the decisions.
        SEED (int): The LLM seed used to generate the decisions.
        TIMESTAMP (str): The timestamp when the decision was made.
        CONTROL_OPTIONS (dict): A dictionary containing the options available for the control template.
        CONTROL_ANSWER (str): The raw decision output from the deciding LLM for the control template.
        CONTROL_DECISION (int): The decision made by the LLM for the control template.
        TREATMENT_OPTIONS (dict): A dictionary containing the options available for the treatment template.
        TREATMENT_ANSWER (str): The raw decision output from the deciding LLM for the treatment template.
        TREATMENT_DECISION (int): The decision made by the LLM for the treatment template.
    """

    def __init__(self, model: str, control_options: dict, control_answer: str, control_decision: int, treatment_options: dict, treatment_answer: str, treatment_decision: int, temperature: float = None, seed: int = None):
        self.MODEL: str = model
        self.TEMPERATURE: float = temperature
        self.SEED: int = seed
        self.TIMESTAMP: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.CONTROL_OPTIONS: dict = control_options
        self.CONTROL_ANSWER: str = control_answer
        self.CONTROL_DECISION: int = control_decision

        self.TREATMENT_OPTIONS: dict = treatment_options
        self.TREATMENT_ANSWER: str = treatment_answer
        self.TREATMENT_DECISION: int = treatment_decision

    def __str__(self) -> str:
        return f"---DecisionResult---\n\nTIMESTAMP: {self.TIMESTAMP}\nMODEL: {self.MODEL}\nTEMPERATURE: {self.TEMPERATURE}\nSEED: {self.SEED}\n\nCONTROL OPTIONS: {self.CONTROL_OPTIONS}\nRAW CONTROL ANSWER: {self.CONTROL_ANSWER}\nCONTROL DECISION: {self.CONTROL_DECISION}\n\nTREATMENT OPTIONS: {self.TREATMENT_OPTIONS}\nRAW TREATMENT ANSWER: {self.TREATMENT_ANSWER}\nTREATMENT DECISION: {self.TREATMENT_DECISION}\n\n------"

    def __repr__(self) -> str:
        return self.__str__()


class Template:
    """
    A class representing a single template (e.g., control or treatment template) for a cognitive bias test case.
    Uses xml.etree.ElementTree internally to store and manipulate the template contents.

    Attributes:
        _data (xml.etree.ElementTree.Element): An Element object storing the contents of this template.
    """
    
    def __init__(self, from_string: str = None, from_file: str = None, from_element: ET.Element = None, type: str = None):
        """
        Instantiates a new Template object. Up to one source (string, file, or element) can be provided. If no source is provided, an empty template will be created.

        Args:
            from_string (str): The XML-like string from which to parse the template.
            from_file (str): The path of the XML file from which to parse the template.
            from_element (xml.etree.ElementTree.Element): The Element object representing the template.
            type (str): The type of the template, either 'control' or 'treatment' (only considered when no source is provided).
        """

        # If more than one sources are given, raise an error
        sources = [from_string, from_file, from_element]        
        if len([source for source in sources if source is not None]) > 1:
            raise ValueError("Only one source can be provided: from_string, from_file, or from_element.")
        
        # Parse the template from the given source
        self._data: ET.Element = None
        if from_string is not None:
            self._data = ET.fromstring(from_string)
            self._validate(allow_incomplete=False)
        elif from_file is not None:
            self._data = ET.parse(from_file).getroot()
            self._validate(allow_incomplete=False)
        elif from_element is not None:
            self._data = from_element
            self._validate(allow_incomplete=False)
        else:
            self._data = ET.Element("template")

            # If a type was given (e.g., 'control' or 'treatment'), store it as an attribute of the root template element
            if type is not None:
                self._data.set("type", type)

    def add_situation(self, situation: str) -> None:
        """
        Adds a situation element to the template.

        Args:
            situation (str): The situation to be added to the template.
        """

        situation_element = ET.Element("situation")
        situation_element.text = situation
        self._data.append(situation_element)
        self._validate(allow_incomplete=True)

    def add_prompt(self, prompt: str) -> None:
        """
        Adds a prompt element to the template.

        Args:
            prompt (str): The prompt to be added to the template.
        """

        prompt_element = ET.Element("prompt")
        prompt_element.text = prompt
        self._data.append(prompt_element)
        self._validate(allow_incomplete=True)

    def add_option(self, option: str) -> None:
        """
        Adds an option element to the template.

        Args:
            option (str): The option to be added to the template.
        """

        option_element = ET.Element("option")
        option_element.text = option
        self._data.append(option_element)
        self._validate(allow_incomplete=True)
    
    def insert(self, pattern: str, text: str, origin: str = None) -> list[dict]:
        """
        Inserts text into a blank in the template based on a pattern to be replaced.
        Text can be user-defined (replacing patterns wrapped by '{{' and '}}') or generated by a model (replacing patterns wrapped by '[[' and ']]').

        Args:
            pattern (str): The pattern to be replaced in the template.
            text (str): The text to replace the pattern.
            origin (str): The origin of the text, either 'user' or 'model'. If specified, only patterns matching the specified origin will be replaced. Otherwise, all matching patterns will be replaced.

        Returns:
            list[dict]: A list of insertions made into the template.
        """

        # Check that a valid origin is provided
        if origin not in ['user', 'model', None]:
            raise ValueError("Origin must be one of 'user', 'model', or None.")

        # If no origin is specified, replace all matching patterns for both origins, 'user' and 'model'
        if origin is None:
            insertions = []
            insertions.extend(self.insert(pattern, text, 'user'))
            insertions.extend(self.insert(pattern, text, 'model'))
            return self._insertions_to_dict(insertions)

        # Prepare the full pattern and text based on the origin
        pattern_full = '{{' + pattern + '}}' if origin == 'user' else '[[' + pattern + ']]'
        text_full = '{{' + text + '}}' if origin == 'user' else '[[' + text + ']]'

        # Iterate over all elements in this template
        for elem in self._data:
            # Skip over all elements that are not of type 'situation', 'prompt', or 'option' (especially 'insertion' elements)
            if elem.tag not in ['situation', 'prompt', 'option']:
                continue

            # Apply all previous insertions to the element's text
            current_text = self._apply_insertions(elem.text, drop_user_brackets=True, drop_model_brackets=True)

            # If the element's text still contains unfilled gaps matching the pattern, accept and return the insertion
            if pattern_full in current_text:
                return self._accept_insertion(pattern, text, origin)

        # If the pattern cannot be found, raise a ValueError
        raise ValueError(f"Could not insert text into template. Pattern {pattern_full} was not found.")

    def insert_values(self, pairs: list[tuple[str, str]], kind: str) -> None:
        """
        This function is deprecated. Use insert instead.
        
        Inserts custom or generated values into the template.
        
        Args:
            pairs (list[tuple[str, str]]): A list of tuples, where each tuple contains a pattern and a value.
            kind (str): The kind of generation for the values being inserted: "manual" or "LLM".   
        """

        warnings.warn("insert_values is deprecated. Use insert instead.", DeprecationWarning)

        if kind == 'manual':
            kind = 'user'
        elif kind == 'LLM':
            kind = 'model'
        else:
            raise ValueError(f"Invalid kind {kind}.")

        for pattern, value in pairs:
            self.insert(pattern, value, kind)

    def get_gaps(self, include_filled: bool = False, include_duplicates: bool = False) -> list[str]:
        """
        Returns a list of all gaps in this template. Gaps are indicated by either {{...}}, to be filled by the user, or [[...]], to be filled by a model.

        Args:
            include_filled (bool): If True, gaps that are already filled (i.e., values were inserted), are also returned.

        Returns:
            list[str]: A list of all gaps in this template.
        """

        def find_gaps(text: str) -> list[str]:
            # Define the regex pattern to match [[...]] or {{...}}
            pattern = r'(\[\[.*?\]\])|(\{\{.*?\}\})'
            
            # Find all matches of the pattern in the string
            matches = re.findall(pattern, text)
            
            # Flatten the list of tuples to get all matched strings. Each element in matches is a tuple, where one of the elements is an empty string
            result = [match[0] if match[0] else match[1] for match in matches]
            
            return result
        
        # Iterate over all elements in this template and find the gaps        
        gaps = []
        for elem in self._data:
            # Skip over all elements that are not of type 'situation', 'prompt', or 'option'
            if elem.tag not in ['situation', 'prompt', 'option']:
                continue

            # Find gaps in the element's text. If include_filled=False, insertions are applied to the text first so that gaps with insertions are not detected
            text = elem.text
            if not include_filled:
                text = self._apply_insertions(text)

            gaps.extend(find_gaps(text))

        # Remove duplicates if requested
        if not include_duplicates:
            gaps = list(dict.fromkeys(gaps))

        return gaps

    def get_insertions(self) -> list[dict]:
        """
        Returns a list of insertions made into the template. Each insertion has three attributes 'origin' (either 'user' or 'model'), 'instruction', and the inserted 'text'.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary represents an insertion.
        """

        insertions = self._data.find("insertions")

        if insertions is None:
            return []

        return self._insertions_to_dict(list(insertions))

    @property
    def inserted_values(self):
        warnings.warn("inserted_values is deprecated. Use the get_insertions function instead.", DeprecationWarning)
        
        insertions = self.get_insertions()
        inserted_values = {}

        for insertion in insertions:
            kind = 'manual' if insertion["origin"] == 'user' else 'LLM'
            inserted_values[insertion["instruction"]] = (insertion["text"], kind)

        return inserted_values
    
    def format(self, insert_headings: bool = True, show_type: bool = False, drop_user_brackets: bool = True, drop_model_brackets: bool = True, shuffle_options: bool = False, seed: int = 42) -> str:
        """
        Formats the template into a string.

        Args:
            insert_headings (bool): Whether to insert headings (Situation, Prompt, Answer Options).
            show_type (bool): Whether to show the type of each element using XML-like tags.
            drop_user_brackets (bool): If True, {{ }} indicating user-made insertions will be removed for every gap with an inserted text.
            drop_model_brackets (bool): If True, [[ ]] indicating model-made insertions will be removed for every gap with an inserted text.
            shuffle_options (bool): If True, answer options will be shuffled randomly using the provided seed. If False, answer options will have the order defined in the template.
            seed (int): The seed used for randomly shuffling answer options. Ignored, if shuffle_options = False.

        Returns:
            str: The formatted string of the template.
        """

        # Validate that the template is complete and not corrupted
        self._validate(allow_incomplete=False)

        # Store the final formatted string in a variable
        formatted = ''

        # Define a function to format an individual element
        def format_element(text: str, type: str) -> str:
            # Fill in the gaps in the element's text according to the insertions made into this template
            text = self._apply_insertions(text, drop_user_brackets=drop_user_brackets, drop_model_brackets=drop_model_brackets)

            if show_type:
                return f'<{type}>{text}</{type}>\n'
            return f'{text}\n'

        # Format all situation elements
        if insert_headings:
            formatted += 'Situation:\n'
        for elem in self._data.findall('situation'):
            formatted += format_element(elem.text, elem.tag)

        # Format all prompt elements
        if insert_headings:
            formatted += '\nPrompt:\n'
        for elem in self._data.findall('prompt'):
            formatted += format_element(elem.text, elem.tag)

        # Format all option elements
        if insert_headings:
            formatted += '\nAnswer Options:\n'
        option_counter = 1
        for option in self.get_options(shuffle_options=shuffle_options, seed=seed)[0]:
            formatted += format_element(f'Option {option_counter}: {option}', 'option')
            option_counter += 1

        return formatted

    def get_options(self, shuffle_options: bool = False, seed: int = 42) -> tuple[list[str], list[int]]:
        """
        Gets the answer options defined in this template and offers functionality to randomly shuffle them.

        Args:
            shuffle_options (bool): If True, the answer options will be randomly shuffled using the provided seed.
            seed (int): The seed to used for shuffling the answer options.

        Returns:
            tuple[list[str], list[int]]: Returns two lists, one with the answer option texts and one with the original zero-based position of the answer option.
        """

        # Get all options defined in this template
        options = self._data.findall('option')
        options = [elem.text for elem in options]

        # Create a list of indices for the options with their original position, i.e., [0, 1, 2, ...]
        indices = list(range(len(options)))

        # If requested, randomly shuffle the options
        if shuffle_options:
            random.Random(seed).shuffle(indices)
        options = [options[i] for i in indices]

        return options, indices

    def _accept_insertion(self, pattern: str, text: str, origin: str) -> list[dict]:
        """
        Stores an insertion into this template.

        Args:
            pattern (str): The pattern replaced by the insertion.
            text (str): The text inserted.
            origin (str): The insertion's origin ('user' or 'model').

        Returns:
            list[dict]: The insertion that was stored.
        """

        # If no insertions have been made so far, create a new insertions element in this template
        if self._data.find("insertions") is None:
            self._data.append(ET.Element("insertions"))

        # Append this insertion to the insertions element
        insertion = ET.Element("insertion", attrib={"origin": origin, "instruction": pattern})
        insertion.text = text
        self._data.find("insertions").append(insertion)
        return self._insertions_to_dict([insertion])

    def _validate(self, allow_incomplete: bool = False) -> bool:
        """
        Validates that this template is complete and not corrupted. Raises a ValueError if corruptions have been found. Otherwise, returns True.

        Args:
            allow_incomplete (bool): Whether to allow the template to be incomplete (i.e., still missing situation, prompt, or option elements).

        Returns:
            bool: True if the template is complete and not corrupted.
        """

        for elem in self._data:
            # Validate that all elements have a valid tag
            if elem.tag not in ['situation', 'prompt', 'option', 'insertions']:
                raise ValueError(f'Templates can only contain elements of type situation, prompt, option, or insertions. Found illegal element of type {elem.tag}.')

            # Validate that situation, prompt, and option elements have no further children
            if elem.tag in ['situation', 'prompt', 'option'] and len(elem) > 0:
                raise ValueError(f'Situation, prompt, and option elements cannot have children. Found a {elem.tag} element with {len(elem)} children.')

            # Validate that all situation, prompt, and option elements have text
            if elem.tag in ['situation', 'prompt', 'option'] and (elem.text is None or elem.text.strip() == ''):
                raise ValueError(f'Situation, prompt, and option elements must not be empty. Found a {elem.tag} element with text "{elem.text}"')

        # Validate that all element types appear in sufficient quantity
        if not allow_incomplete:
            if len(self._data.findall('situation')) == 0:
                raise ValueError('The template must contain at least one situation element.')
            if len(self._data.findall('prompt')) != 1:
                raise ValueError(f'The template must contain exactly one prompt element. Found {len(self._data.findall('prompt'))}.')
            if len(self._data.findall('option')) < 2:
                raise ValueError(f'The template must contain at least two option elements. Found {len(self._data.findall('option'))}.')

        # Validate that situation, prompt, and option elements appear strictly in that order
        last = None
        for elem in self._data:
            if elem.tag == 'insertions':
                # Ignore insertion elements as they are invisible to the user
                continue
            if elem.tag == 'situation':
                if last not in [None, 'situation']:
                    raise ValueError(f"Situation elements cannot follow {last} elements. Situation elements must be first in a template.")
            elif elem.tag == 'prompt':
                if last not in [None, 'situation']:
                    raise ValueError(f"Prompt elements must directly follow situation elements. Found a prompt element following a {last} element.")
            elif elem.tag == 'option':
                if last not in [None, 'prompt', 'option']:
                    raise ValueError(f"Option elements must directly follow prompt or other option elements. Found an option element following a {last} element.")
            last = elem.tag

        # Validate that insertions have the correct format
        insertions = self._data.findall('insertions')
        if len(insertions) > 1:
            raise ValueError(f"There can only be one insertions element in a template. Found {len(insertions)}.")
        if len(insertions) == 1:
            for elem in insertions[0]:
                if elem.tag != 'insertion':
                    raise ValueError(f"The insertions element must contain only insertion elements. Found a {elem.tag} element inside the insertions element.")
                if 'origin' not in elem.attrib:
                    raise ValueError(f"Insertion elements must have an origin attribute. Found an insertion element without origin.")
                if elem.attrib['origin'] not in ['user', 'model']:
                    raise ValueError(f"The origin of an insertion element must be either 'user' or 'model'. Found origin {elem.attrib['origin']}.")
                if 'instruction' not in elem.attrib:
                    raise ValueError(f"Insertion elements must have an instruction attribute. Found an insertion element without instruction.")
                if elem.attrib['instruction'].strip() in [None, '']:
                    raise ValueError(f"The instruction of an insertion element must not be empty. Found empty instruction '{elem.attrib['instruction']}'.")

        return True

    def _apply_insertions(self, text: str, drop_user_brackets: bool = True, drop_model_brackets: bool = True) -> str:
        """
        Applies all insertions that were made into this template to the provided text.

        Args:
            text (str): The text to which to apply the insertions.
            drop_user_brackets (bool): If True, {{ }} indicating user-made insertions will be removed for every gap with an inserted text.
            drop_model_brackets (bool): If True, [[ ]] indicating model-made insertions will be removed for every gap with an inserted text.

        Returns:
            str: The adjusted text where insertions are made into the indicated gaps.
        """

        # Get the insertions that were made into this template
        insertions = self.get_insertions()

        # Iterate over all insertions and apply them to the text
        for insertion in insertions:
            pattern = insertion['instruction']
            insertion_text = insertion['text']
            origin = insertion['origin']

            # Expand the search pattern and inserted text with the respective brackets where needed
            if origin == 'user':
                pattern = '{{' + pattern + '}}'
                if not drop_user_brackets:
                    insertion_text = '{{' + insertion_text + '}}'
            elif origin == 'model':
                pattern = '[[' + pattern + ']]'
                if not drop_model_brackets:
                    insertion_text = '[[' + insertion_text + ']]'

            # Apply the insertion to the text
            text = text.replace(pattern, insertion_text)

        return text

    def _insertions_to_dict(self, insertions: list[ET.Element]) -> list[dict]:
        """
        Converts a list of xml.etree.ElementTree.Element objects representing insertions into a list of dictionaries.

        Args:
            insertions (list[xml.etree.ElementTree.Element]): A list of Element objects representing insertions.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary represents an insertion.
        """

        insertions_list = []
        for insertion in insertions:
            insertions_list.append({
                "origin": insertion.attrib["origin"],
                "instruction": insertion.attrib["instruction"],
                "text": insertion.text
            })
        
        return insertions_list

    def __str__(self) -> str:
        return self.format(insert_headings=True, show_type=False)

    def __repr__(self) -> str:
        return self.format(insert_headings=False, show_type=True)


class TestConfig:
    """
    A class representing a configuration file for a cognitive bias test.

    Attributes:
        config (xml.etree.ElementTree.ElementTree): An ElementTree object representing the XML configuration file.
    """

    def __init__(self, path: str):
        """
        Instantiates a new TestConfig object.

        Args:
            path (str): The path to the XML configuration file.
        """

        self.config = self._load(path)

    def get_bias_name(self) -> str:
        """
        Returns the name of the cognitive bias being tested.

        Returns:
            str: The name of the cognitive bias being tested.
        """

        return self.config.getroot().get('bias')
    
    def get_custom_values(self) -> dict:
        """
        Returns the custom values defined in the configuration file.

        Returns:
            dict: A dictionary containing the custom values defined in the configuration file.
        """

        custom_values = self.config.getroot().findall('custom_values')
        custom_values_dict = {}
        for custom_value in custom_values:
            key = custom_value.get('name')
            if len(custom_value) == 0:
                custom_values_dict[key] = None
            elif len(custom_value) == 1:
                custom_values_dict[key] = custom_value.find("value").text
            else:
                custom_values_dict[key] = [value.text for value in custom_value]
        
        return custom_values_dict

    def get_variants(self) -> list[str]:
        """
        Returns a list of variant names defined in the configuration file.

        Returns:
            A list of variant names defined in the configuration file.
        """

        # Find all variant elements in the configuration file
        variants = self.config.getroot().findall('variant')

        # Return the names of all variants
        return [variant.get('name') for variant in variants]

    def get_template(self, template_type: str = "control", variant: str = None) -> Template:
        """
        Returns a template from the test configuration.

        Args:
            template_type (str): The type of the template ('control' or 'treatment').
            variant (str): The name of the variant. Only needed if the test configuration includes multiple variants.

        Returns:
            Template: A Template object representing the template.
        """

        root = self.config.getroot()

        if variant is not None:
            # Find the variant element with the specified name
            variant_element = root.find(f"variant[@name='{variant}']")
            if variant_element is None:
                raise ValueError(f"Variant '{variant}' not found in the configuration.")
        else:
            # No variant was specified, try to find the next best variant
            found_variants = root.findall('variant')
            if len(found_variants) > 1:
                raise ValueError(f"{len(found_variants)} variants found in the configuration. Please specify in which variant to find the template.")
            elif len(found_variants) == 1:
                variant_element = found_variants[0]
            else:
                # No variant elements in the configuration file, treat the root as the variant element
                variant_element = root

        # Find the template with the specified type
        template_config = variant_element.find(f"template[@type='{template_type}']")
        if template_config is None:
            raise ValueError(f"No template with type '{template_type}' found in variant '{variant}'.")

        # Parse the template
        template = Template(from_element=template_config)

        return template

    def get_control_template(self, variant: str = None) -> Template:
        """
        Returns the control template for the specified variant or the Default variant if none is specified.

        Args:
            variant (str): The name of the variant.

        Returns:
            Template: A Template object representing the control template.
        """

        return self.get_template("control", variant)

    def get_treatment_template(self, variant: str = None) -> Template:
        """
        Returns the treatment template for the specified variant or the Default variant if none is specified.

        Args:
            variant (str): The name of the variant.

        Returns:
            Template: A Template object representing the treatment template.
        """

        return self.get_template("treatment", variant)
    
    def _load(self, path: str) -> ET.ElementTree:
        """
        Loads the XML configuration file for the specified cognitive bias.

        Args:
            path (str): The path to the XML configuration file.

        Returns:
            An xml.etree.ElementTree.ElementTree object representing the XML configuration file.
        """

        return ET.parse(path)


class TestCase:
    """
    A class representing a cognitive bias test case.

    Attributes:
        BIAS (str): The name of the cognitive bias being tested.
        CONTROL (Template): The control template for the test case.
        TREATMENT (Template): The treatment template for the test case.
        GENERATOR (str): The name of the LLM generator used to populate the templates.
        SCENARIO (str): The scenario in which the test case is being conducted.
        VARIANT (str, optional): The variant of the test case.
        REMARKS (str, optional): Any additional remarks about the test case.
    """

    def __init__(self, bias: str, control: Template, treatment: Template, generator: str, scenario: str, variant: str = None, remarks: str = None, **kwargs):
        self.BIAS: str = bias
        self.CONTROL: Template = control
        self.TREATMENT: Template = treatment
        self.GENERATOR: str = generator
        self.SCENARIO: str = scenario
        self.VARIANT: str = variant
        self.REMARKS: str = remarks

        # Issue a deprecation warning for fields that are no longer supported
        for key, value in kwargs.items():
            if key in ['control_values', 'treatment_values']:
                warnings.warn("'control_values' and 'treatment_values' are deprecated and should not be used anymore. All inserted values are automatically stored inside the Template objects, which in turn are stored in TestCase.CONTROL and TestCase.Treatment. If these values are important for understanding which test was run, consider using the TestCase.VARIANT attribute to keep track of these values.", DeprecationWarning)

            setattr(self, key.upper(), value)

    def __str__(self) -> str:
        return f'---TestCase---\n\nBIAS: {self.BIAS}\nVARIANT: {self.VARIANT}\nSCENARIO: {self.SCENARIO}\nGENERATOR: {self.GENERATOR}\n\nCONTROL:\n{self.CONTROL}\n\nTREATMENT:\n{self.TREATMENT}\n\nREMARKS:\n{self.REMARKS}\n\n------'

    def __repr__(self) -> str:
        return self.__str__()