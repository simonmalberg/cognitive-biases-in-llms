import generation as gen
from models import RandomModel
import random


if __name__ == "__main__":

    # Load the pre-defined scenario strings
    with open('scenarios.txt') as f:
        scenarios = f.readlines()
    
    # Format the scenario strings by removing any markdown
    scenarios = [s.strip().replace('**', '') for s in scenarios]

    # Randomly pick a scenario
    scenario = random.choice(scenarios)

    # Generate a dummy test case and print it 
    model = RandomModel()
    generator = gen.DummyTestGenerator()
    test_case = generator.generate(model, scenario)
    print(test_case)