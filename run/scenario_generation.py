import sys
import os

# Add the project root directory to sys.path to be able to import functionality from core/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.utils import get_model
import json
import argparse


# References to relevant data directories
DATA_DIRECTORY = os.path.join(".", "data")
SCENARIOS_FILE = "scenarios_test.txt"


# Taxonomy of industry groups as per the Global Industry Classification Standard (GICS)
INDUSTRY_GROUPS = [
    "Energy",
    "Materials",
    "Capital Goods",
    "Commercial & Professional Services",
    "Transportation",
    "Automobiles & Components",
    "Consumer Durables & Apparel",
    "Consumer Services",
    "Consumer Discretionary Distribution & Retail",
    "Consumer Staples Distribution & Retail",
    "Food, Beverage & Tobacco",
    "Household & Personal Products",
    "Health Care Equipment & Services",
    "Pharmaceuticals, Biotechnology & Life Sciences",
    "Banks",
    "Financial Services",
    "Insurance",
    "Software & Services",
    "Technology Hardware & Equipment",
    "Semiconductors & Semiconductor Equipment",
    "Telecommunication Services",
    "Media & Entertainment",
    "Utilities",
    "Equity Real Estate Investment Trusts",
    "Real Estate Management & Development",
]


def get_positions(model_name: str, industry_groups: list[str], n: int = 8, temperature: float = 0.7, seed: int = 42) -> dict:
    """
    Get n managerial positions in the industry group using the GPT-4o model.

    Args:
        model_name (str): The name of the LLM to use.
        industry_groups (list[str]): the list of industry groups
        n (int): the number of managerial positions to generate for each industry group
        temperature (float): the temperature to use in the model for generating the positions
        seed (int): the seed to use for generating the positions
    
    Returns:
        dict: A dictionary containing the industry groups as keys and the list of positions as values
    """

    # Get an instance of the model
    model = get_model(model_name)

    # Prepare the prompt
    user_prompt = model._PROMPTS["position_generation_prompt"]
    user_prompt = user_prompt.replace("{{n}}", str(n))
    user_prompt = user_prompt.replace("{{industry_groups}}", str(industry_groups))

    # Run the prompt
    response = model._CLIENT.chat.completions.create(
        model=model.NAME,
        temperature=temperature,
        seed=seed,
        messages=[{"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"},
    )
    
    positions = json.loads(response.choices[0].message.content)

    return positions


def get_scenarios(model_name: str, positions: dict, temperature: float = 0.7, seed: int = 42) -> list[str]:
    """
    Get scenarios for the given managerial positions using a model.
    
    Args:
        model_name (str): The name of the LLM to use.
        positions (dict): a dictionary containing the industry groups as keys and the list of positions as values
        temperature (float): the temperature to use in the model for generating the scenarios
        seed (int): the seed to use for generating the scenarios
    
    Returns:
        list[str]: A list of generated scenarios
    """

    # Get an instance of the model
    model = get_model(model_name)

    all_scenarios = []

    # Generate scenarios for each industry group
    for industry, managers in positions.items():
        scenarios = []
        user_prompt = model._PROMPTS["scenario_generation_prompt"]

        # Convert {industry_group: [positions]} to a required sentence format
        scenarios.append(
            [
                f"{{A/An}} {manager.lower()} at a company from the {industry.lower()} industry deciding {{decision}}."
                for manager in managers
            ]
        )
        user_prompt = user_prompt.replace("{{scenarios}}", str(scenarios))
        response = model._CLIENT.chat.completions.create(
            model=model.NAME,
            temperature=temperature,
            seed=seed,
            messages=[{"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
        )

        scenarios = json.loads(response.choices[0].message.content)
        all_scenarios += list(scenarios.values())

    return all_scenarios


def main():
    """
    The main function of this script that generates scenario strings.
    """

    # Define a command line argument parser
    parser = argparse.ArgumentParser(description="This script generates scenario strings.")
    parser.add_argument("--model", type=str, help="The LLM to use for sampling the scenario strings.", default="GPT-4o")
    parser.add_argument("--file_name", type=str, help="The name of the file to write the scenario strings to.", default=SCENARIOS_FILE)
    parser.add_argument("--industry_id", type=int, help="The index of the industry group from the Global Industry Classification Standard (GICS), ranging from 0 to 24. If -1, the script will iterate over all 25 industry groups.", default=-1)
    parser.add_argument("--n_positions", type=int, help="The number of manager positions to be sampled for each industry group.", default=8)
    parser.add_argument("--temperature", type=float, help="The LLM's temperature parameter to use for sampling the scenarios.", default=1.0)
    parser.add_argument("--seed", type=int, help="The seed to use for sampling the scenarios.", default=42)
    args = parser.parse_args()

    # Valdiate the industry group index
    if args.industry_id < -1 or args.industry_id >= len(INDUSTRY_GROUPS):
        raise ValueError(f"Argument industry_id must be an integer in range [-1, 24] but is set to {args.industry_id}.")

    # Assemble the path to the scenarios file
    file_path = os.path.join(DATA_DIRECTORY, args.file_name)

    # Assemble a list of indices of all industry groups to sample scenario strings for
    industry_groups = list(range(len(INDUSTRY_GROUPS))) if args.industry_id == -1 else [args.industry_id]

    # Sample scenario strings for all selected industry groups
    for i in industry_groups:
        # Sample n managerial positions common for the industry group
        print(f"Get positions for: {INDUSTRY_GROUPS[i]}")
        positions = get_positions(args.model, [INDUSTRY_GROUPS[i]], n=args.n_positions, temperature=args.temperature, seed=args.seed)

        # Sample one scenario string for each managerial position in the industry group
        print(f"Get scenarios for: {positions}")
        scenarios = get_scenarios(args.model, positions, temperature=args.temperature, seed=args.seed)

        # Write the scenario strings to a file
        with open(file_path, "a", encoding="utf-8") as f:
            for scenario in scenarios:
                f.write(scenario + "\n")

    # Remove all empty lines from the file (the file has one empty line at the very end)
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(file_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.strip():  # Check if the line is not empty after stripping whitespace
                f.write(line)


if __name__ == "__main__":
    main()