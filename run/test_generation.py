import sys
import os

# Add the project root directory to sys.path to be able to import functionality from core/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.utils import get_generator, get_model
from core.base import TestCase, Template
from collections import defaultdict
from tqdm import tqdm
from xml.dom import minidom
import xml.etree.ElementTree as ET
import pandas as pd
import hashlib
import argparse


# References to relevant data directories
SCENARIOS_PATH = os.path.join(".", "data", "scenarios.txt")
TEST_CASES_PATH = os.path.join(".", "data", "generated_tests")
GENERATION_LOGS_PATH = os.path.join(".", "data", "generation_logs")
DATASETS_PATH = os.path.join(".", "data", "generated_datasets")
TESTS_PATH = os.path.join(".", "tests")


def get_all_biases() -> list[str]:
    """
    Retrieves a list of all cognitive biases implemented in the repository.

    Returns:
        list[str]: A list of cognitive biases.
    """

    return [dir for dir in os.listdir(TESTS_PATH) if dir.isalpha()]


def prettify_xml(elem: ET.Element) -> str:
    """
    Returns a pretty-printed XML string for the Element, with our custom formatting.

    Args:
        elem (ET.Element): The ElementTree element to pretty-print.

    Returns:
        str: A pretty-printed XML string of the element.
    """

    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")
    lines = [line for line in pretty_xml.split("\n") if line.strip()]  # Remove empty lines
    return "\n".join(lines)


def write_to_xml(test_cases: list[TestCase], dir_name: str, file_name: str):
    """
    Writes the test case instances to an XML file.

    Args:
        test_cases (list[TestCase]): A list of test case instances to write to the XML file.
        dir_name (str): The name of the directory to write the test case instances to.
        file_name (str): The name of the file to write the test case instances to (must end with .xml).
    """

    # Create the directory, if it does not yet exist
    os.makedirs(dir_name, exist_ok=True)

    # Join the directory and file names into the full file path
    file_path = os.path.join(dir_name, file_name)

    with open(file_path, "wb") as output:
        # Write the dataset root element
        output.write(b"<dataset>")

        # Write the test case instances to the XML file
        for test_case in test_cases:
            output.write(b"<test_case>")
            output.write(("<bias>" + str(test_case.BIAS) + "</bias>\n").encode())
            output.write(("<variant>" + str(test_case.VARIANT) + "</variant>\n").encode())
            output.write(("<scenario>" + str(test_case.SCENARIO.replace("&", "&amp;")) + "</scenario>\n").encode())
            output.write(("<generator>" + str(test_case.GENERATOR) + "</generator>\n").encode())
            output.write(("<temperature>" + str(test_case.TEMPERATURE) + "</temperature>\n").encode())
            output.write(("<timestamp>" + str(test_case.TIMESTAMP) + "</timestamp>\n").encode())
            output.write(("<seed>" + str(test_case.SEED) + "</seed>\n").encode())
            ET.ElementTree(test_case.CONTROL._data).write(output, encoding="utf-8")
            ET.ElementTree(test_case.TREATMENT._data).write(output, encoding="utf-8")
            output.write(("<remarks>" + str(test_case.REMARKS) + "</remarks>\n").encode())
            output.write(b"</test_case>")

        output.write(b"</dataset>")

    # Prettify the XML file and write it back to the file
    pretty_xml = prettify_xml(ET.parse(file_path).getroot())
    with open(file_path, "w") as f:
        f.write(pretty_xml)


def write_to_txt(test_cases: list[TestCase], dir_name: str, file_name: str):
    """
    Writes the test case instances to a TXT file.

    Args:
        test_cases (list[TestCase]): A list of test case instances to write to the TXT file.
        dir_name (str): The name of the directory to write the test case instances to.
        file_name (str): The name of the file to write the test case instances to (must end with .txt).
    """

    # Create the directory, if it does not yet exist
    os.makedirs(dir_name, exist_ok=True)

    # Write all the test case instances to a .txt file
    with open(os.path.join(dir_name, file_name), "a+") as f:
        for test_case in test_cases:
            f.write(f"{test_case}\n")


def assemble_dataset(tests_directory: str, output_directory: str, file_name: str):
    """
    Assembles and stores a dataset in CSV format from the generated test case instances in the directory.

    Args:
        tests_directory (str): The name of the directory where the generated XML test case instances are stored.
        output_directory (str): The name of the directory where the dataset should be stored.
        file_name (str): The name of the file to save the dataset as (must end with .csv).
    """

    # Create the directory, if it does not yet exist
    os.makedirs(output_directory, exist_ok=True)

    # Join the directory and file names into the full file path
    file_path = os.path.join(output_directory, file_name)

    # Create a pandas DataFrame to store all test case instances
    dataset = pd.DataFrame()

    # Iterate over all XML files with generated test case instances
    for directory, _, files in tqdm(os.walk(tests_directory)):
        for file in files:
            if file.endswith(".xml"):
                # Load the XML file
                tree = ET.parse(os.path.join(directory, file))
                root = tree.getroot()

                # Create a dictionary to store the test case instances
                tests = defaultdict(list)

                # Parse all test case instances from the XML file
                for test_case in root:
                    # Keep the raw XML templates in string format to be able to easily load them during testing
                    control_template = test_case.find("./template[@type='control']")
                    treatment_template = test_case.find("./template[@type='treatment']")
                    tests["raw_control"].append(
                        ET.tostring(control_template, encoding="unicode", method="xml")
                    )
                    tests["raw_treatment"].append(
                        ET.tostring(treatment_template, encoding="unicode", method="xml")
                    )

                    # Parse all other components of the test case instance
                    for child in test_case:
                        if child.tag != "template":
                            tests[child.tag].append(child.text)
                        else:
                            t = Template(from_element=child)
                            tests[child.attrib["type"]].append(t.format())

                # Append the parsed test case instances to the dataset
                temp_df = pd.DataFrame(tests)
                dataset = pd.concat([dataset, temp_df], ignore_index=True)

    # Save the dataset as a CSV file
    dataset.to_csv(file_path, index=False)

    print(f"Dataset is successfully assembled and saved in {file_path}")


def generate_test_cases(biases: list[str], model: str, scenarios: list[str], temperature: float = 0.0, num_instances: int = 5, max_retries: int = 5, seed: int = 0):
    """
    Generates a dataset of test case instances for provided biases.

    Args:
        biases (list[str]): A list of cognitive biases to generate the dataset for.
        model (str): The name of the model to use for generating the test case instances.
        scenarios (list[str]): A list of scenarios to generate the test case instances for.
        temperature (float): The temperature of the LLM to use for generating the test case instances.
        num_instances (int): The number of instances to generate for each scenario.
        max_retries (int): The maximum number of retries per instance in case of errors.
        seed (int): The starting seed to use for generating the test case instances.
    """

    # Iterate over all passed cognitive biases and generate test case instances for them
    for bias in biases:
        print(f"Start generation of test case instances for bias: {bias}")

        # Retrieve the test generator for that cognitive bias and the generation model selected by the user
        generator = get_generator(bias)
        generation_model = get_model(model)

        # Iterate over all passed scenarios and generate num_instances test case instances per scenario
        for scenario in tqdm(scenarios):
            test_cases: list[TestCase] = generator.generate_all(
                generation_model,
                [scenario],
                temperature,
                seed,
                num_instances,
                max_retries,
            )

            # Save the generated test cases
            file_name = f"{int(hashlib.md5(scenario.encode()).hexdigest(), 16)}_{num_instances}"     # The filename is a hash of the scenario followed by _{num_instances}
            write_to_xml(test_cases, os.path.join(TEST_CASES_PATH, bias), f"{file_name}.xml")        # Store raw test case instances in an XML
            write_to_txt(test_cases, os.path.join(GENERATION_LOGS_PATH, bias), f"{file_name}.txt")   # Store formatted test case instances in a TXT for user-friendly logging


def main():
    """
    The main function of this script that parses the command line arguments and starts the dataset generation.
    """

    # Define a command line argument parser
    parser = argparse.ArgumentParser(description="This script generates test case instances.")
    parser.add_argument("--bias", type=str, help="The names of the cognitive biases to generate test cases for. Separate multiple biases with a comma. If not provided, this will default to all cognitive biases defined in this repository.", default=None)
    parser.add_argument("--model", type=str, help="The LLM to use as generation model.", default="GPT-4o")
    parser.add_argument("--scenarios", type=str, help="Path to a file storing scenario strings.", default=SCENARIOS_PATH)
    parser.add_argument("--temperature", type=float, help="Temperature value of the generation LLM", default=0.7)
    parser.add_argument("--num_instances", type=int, help="Number of test case instances to generate per scenario.", default=5)
    parser.add_argument("--max_retries", type=int, help="The maximum number of retries in case of LLM/API-related errors.", default=1)
    parser.add_argument("--seed", type=int, help="The seed to use for the reproducibility.", default=0)
    args = parser.parse_args()

    # Parse the list of selected biases to generate test case instances for. If none are provided, select all biases implemented in this repository
    all_biases = get_all_biases()
    biases = []
    if args.bias is not None:
        biases = [b.strip() for b in args.bias.split(',')]
        biases = [b.title().replace(' ', '') if ' ' in b else b for b in biases]
    if len(biases) == 0:
        biases = all_biases

    # Validate that all selected biases are supported
    for bias in biases:
        if bias not in all_biases:
            raise ValueError(f"Unknown bias '{bias}'. Only the following biases are supported: {all_biases}")

    # Load the scenario strings
    with open(args.scenarios) as f:
        scenarios = f.readlines()
    scenarios = [s.strip() for s in scenarios]

    # Generate the test case instances
    generate_test_cases(
        biases=biases,
        model=args.model,
        scenarios=scenarios,
        temperature=args.temperature,
        num_instances=args.num_instances,
        max_retries=args.max_retries
    )

    # Assemble one dataset per bias from the generated test case instances
    for bias in biases:
        assemble_dataset(tests_directory=os.path.join(TEST_CASES_PATH, bias), output_directory=DATASETS_PATH, file_name=f"{bias}_dataset.csv")


if __name__ == "__main__":
    main()