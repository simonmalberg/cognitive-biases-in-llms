from utils import get_generator, get_model
from base import TestCase
import random
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm


def all_biases() -> list[str]:
    """
    Retrieves a list of all cognitive biases implemented in the repository.

    Returns:
        list[str]: A list of cognitive biases.
    """
    return [dir for dir in os.listdir("biases") if dir.isalpha()]


def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element, with our custom formatting."""
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(
        indent="    "
    )  # 4 spaces indentation for nested elements
    # Remove extra newlines from minidom output
    lines = [
        line for line in pretty_xml.split("\n") if line.strip()
    ]  # Remove empty lines
    return "\n".join(lines)


def write_to_xml(test_cases: list[TestCase], file_path: str) -> None:
    """
    Writes the test cases to an XML file.

    Args:
        test_cases (list[TestCase]): A list of test cases to write to the XML file.
        file_path (str): The path to the file to write the test cases to.
    """

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

    print(f"The test cases are successfully saved to the file {file_path}")


def dataset_generation(
    biases: list[str],
    population_model: str,
    scenarios: list[str],
    file_path: str,
    temperature: float = 0.0,
    num_instances: int = 5,
    max_retries: int = 5,
) -> dict[str, list[TestCase]]:
    """
    Generates a dataset of test cases for provided biases.

    Args:
        biases (list[str]): A list of cognitive biases to generate the dataset for.
        population_model (str): The name of the population model to use for generating the test cases.
        scenarios (list[str]): A list of scenarios to generate the test cases for.
        file_path (str): The path to the file to save the XML file to.
        temperature (float): The temperature of the LLM to use for generating the test cases.
        num_instances (int): The number of instances to generate for each scenario.
        max_retries (int): The maximum number of retries in generation of all tests for a single bias.
    """
    seed: int = 0
    test_cases: list[TestCase] = []
    for bias in biases:
        generator = get_generator(bias)
        for scenario in tqdm(scenarios):
            test_cases += generator.generate_all(
                get_model(population_model),
                [scenario],
                temperature,
                seed,
                num_instances,
                max_retries,
            )
    # Save the dataset to an XML file
    write_to_xml(test_cases, file_path)
    print("The dataset is successfully created.")

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
        scenarios=random.sample(scenarios, 2),
        file_path="dataset_test.xml",
        temperature=0.7,
        num_instances=2,
        max_retries=5,
    )
