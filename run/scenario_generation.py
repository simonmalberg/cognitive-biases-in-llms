from core.utils import get_model
import json

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


def get_positions(
    industry_groups: list[str], n: int = 8, temperature: float = 0.7, seed: int = 42
) -> dict:
    """
    Get n managerial positions in the industry group using the GPT-4o model.

    Args:
    industry_groups (list[str]): the list of industry groups
    n (int): the number of managerial positions to generate for each industry group
    temperature (float): the temperature to use in the model for generating the positions
    seed (int): the seed to use for generating the positions
    
    Returns:
        dict: A dictionary containing the industry groups as keys and the list of positions as values
    """
    model = get_model("GPT-4o")
    user_prompt = model._PROMPTS["position_generation_prompt"]
    user_prompt = user_prompt.replace("{{n}}", str(n))
    user_prompt = user_prompt.replace("{{industry_groups}}", str(industry_groups))
    response = model._CLIENT.chat.completions.create(
        model=model.NAME,
        temperature=temperature,
        seed=seed,
        messages=[{"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"},
    )
    positions = json.loads(response.choices[0].message.content)

    return positions


def get_scenarios(positions: dict, temperature: float = 0.7, seed: int = 42) -> list[str]:
    """
    Get scenarios for the given managerial positions using a model.
    
    Args:
    positions (dict): a dictionary containing the industry groups as keys and the list of positions as values
    temperature (float): the temperature to use in the model for generating the scenarios
    seed (int): the seed to use for generating the scenarios
    
    Returns:
        list[str]: A list of generated scenarios
    """
    model = get_model("GPT-4o")
    all_scenarios = []
    # generating scenarios for each industry group
    for industry, managers in positions.items():
        scenarios = []
        user_prompt = model._PROMPTS["scenario_generation_prompt"]
        # converting {industry_group: [positions]} to a required sentence format
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


if __name__ == "__main__":

    positions = get_positions([INDUSTRY_GROUPS[0]], n=8, temperature=1, seed=42)
    scenarios = get_scenarios(positions, temperature=1, seed=42)
    # write the scenarios to a file
    with open("scenarios_new.txt", "w") as f:
        for scenario in scenarios[:-1]:
            f.write(scenario + "\n")
        f.write(scenarios[-1])