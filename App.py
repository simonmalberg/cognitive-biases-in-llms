import generation as gen
from models import RandomModel
import random

# currently supported biases
BIASES = ['AnchoringBias', 'HaloEffect', 'LossAversion']


if __name__ == "__main__":

    # Load the pre-defined scenario strings
    with open('scenarios.txt') as f:
        scenarios = f.readlines()
    
    # Format the scenario strings by removing any markdown
    scenarios = [s.strip().replace('**', '') for s in scenarios]

    # Randomly pick a scenario
    scenario = random.choice(scenarios)

    # Randomly pick a cognitive bias
    bias = random.choice(BIASES) # TODO: come up with an approach to store all biases' names
                                 # TODO: load bias from command line arguments

    # Generate a dummy test case and print it 
    model = RandomModel()
    # Load the respective test generator for bias
    generator = gen.get_generator(bias)
    test_case = generator.generate(model, scenario)
    print(test_case)