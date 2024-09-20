from utils import get_generator, get_model
from base import TestCase, Template
import random
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from xml.dom import minidom
from tqdm import tqdm
import pandas as pd


def all_biases() -> list[str]:
    """
    Retrieves a list of all cognitive biases implemented in the repository.

    Returns:
        list[str]: A list of cognitive biases.
    """
    return [dir for dir in os.listdir("biases") if dir.isalpha()]


def prettify_xml(elem: ET.Element) -> str:
    """
    Return a pretty-printed XML string for the Element, with our custom formatting.
    """
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")
    lines = [
        line for line in pretty_xml.split("\n") if line.strip()
    ]  # Remove empty lines
    return "\n".join(lines)


def write_to_xml(test_cases: list[TestCase], dir_name: str, file_name: str) -> int:
    """
    Writes the test cases to an XML file.

    Args:
        test_cases (list[TestCase]): A list of test cases to write to the XML file.
        dir_name (str): The name of the directory to write the test cases to.
        file_name (str): The name to the file to write the test cases to.
    """

    os.makedirs(dir_name, exist_ok=True)
    file_path = os.path.join(dir_name, file_name)

    with open(file_path, "wb") as output:
        # Write the dataset root element
        output.write(b"<dataset>")
        # Write the test cases to the XML file
        for test_case in test_cases:
            output.write(b"<test_case>")
            output.write(("<bias>" + test_case.BIAS + "</bias>\n").encode())
            output.write(
                ("<variant>" + str(test_case.VARIANT) + "</variant>\n").encode()
            )
            output.write(("<scenario>" + test_case.SCENARIO + "</scenario>\n").encode())
            output.write(
                ("<generator>" + test_case.GENERATOR + "</generator>\n").encode()
            )
            output.write(
                (
                    "<temperature>" + str(test_case.TEMPERATURE) + "</temperature>\n"
                ).encode()
            )
            output.write(("<seed>" + str(test_case.SEED) + "</seed>\n").encode())
            ET.ElementTree(test_case.CONTROL._data).write(output, encoding="utf-8")
            ET.ElementTree(test_case.TREATMENT._data).write(output, encoding="utf-8")
            output.write(
                ("<remarks>" + str(test_case.REMARKS) + "</remarks>\n").encode()
            )
            output.write(b"</test_case>")
        output.write(b"</dataset>")

    # Prettify the XML file and write it back to the file
    pretty_xml = prettify_xml(ET.parse(file_path).getroot())
    with open(file_path, "w") as f:
        f.write(pretty_xml)

    return 0


def dataset_assemble(dir: str = "generated_tests", save_as: str = "dataset.csv") -> int:
    """
    Assembles the dataset from the generated tests in the directory

    Args:
        dir (str): the directory where the .xml tests are stored. The folder should be in the same directory as the script.
        save_as (str): the name of the file to save the dataset as. The file will be saved in the same directory as the script.
    """

    dataset = pd.DataFrame()

    for file in tqdm(os.listdir(dir)):
        tree = ET.parse(os.path.join(dir, file))
        root = tree.getroot()

        # a dictionary to store the tests
        tests = defaultdict(list)

        # iterating over tests
        for test_case in root:
            # saving the templates in the raw string to be easily loaded in the testing
            control_template = test_case.find("./template[@type='control']")
            treatment_template = test_case.find("./template[@type='treatment']")
            tests["raw_control"].append(
                ET.tostring(control_template, encoding="unicode", method="xml")
            )
            tests["raw_treatment"].append(
                ET.tostring(treatment_template, encoding="unicode", method="xml")
            )
            # iterating over compotents of a test
            for child in test_case:
                # saving each component of the test to the dictionary
                if child.tag != "template":
                    tests[child.tag].append(child.text)
                else:
                    t = Template(from_element=child)
                    tests[child.attrib["type"]].append(t.format())

        # creating a temporary DataFrame from the dictionary
        temp_df = pd.DataFrame(tests)
        # concating the temporary DataFrame to the dataset
        dataset = pd.concat([dataset, temp_df], ignore_index=True)

    # saving the dataset
    dataset.to_csv(save_as, index=False)

    print(f"Dataset is successfully assembled and saved in {save_as}")

    return 0


def dataset_generation(
    biases: list[str],
    population_model: str,
    scenarios: list[str],
    temperature: float = 0.0,
    num_instances: int = 5,
    max_retries: int = 5,
) -> int:
    """
    Generates a dataset of test cases for provided biases.

    Args:
        biases (list[str]): A list of cognitive biases to generate the dataset for.
        population_model (str): The name of the population model to use for generating the test cases.
        scenarios (list[str]): A list of scenarios to generate the test cases for.
        temperature (float): The temperature of the LLM to use for generating the test cases.
        num_instances (int): The number of instances to generate for each scenario.
        max_retries (int): The maximum number of retries in generation of all tests for a single bias.
    """
    seed: int = 0
    test_cases: list[TestCase] = []
    for bias in biases:
        generator = get_generator(bias)
        print(f"Start generation of test cases for bias: {bias}")
        for scenario in tqdm(scenarios):
            test_cases = generator.generate_all(
                get_model(population_model),
                [scenario],
                temperature,
                seed,
                num_instances,
                max_retries,
            )
            # Save the generated test cases for given bias and scenario to an XML file
            _ = write_to_xml(
                test_cases,
                "generated_tests",
                f"{bias}_{hash(scenario)}_{num_instances}.xml",
            )
    # Assemble the dataset from the generated tests
    _ = dataset_assemble()

    return 0


if __name__ == "__main__":

    # Load the pre-defined scenario strings
    with open("scenarios.txt") as f:
        scenarios = f.readlines()

    # Format the scenario strings by removing any markdown
    scenarios = [s.strip().replace("**", "") for s in scenarios]
    model = "GPT-4o"

    dataset_generation(
        biases=["EndowmentEffect", "FramingEffect"],
        population_model=model,
        scenarios=random.sample(scenarios, 3),
        temperature=0.7,
        num_instances=2,
        max_retries=5,
    )
