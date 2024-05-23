import generation as gen
from models import RandomModel
import random
import yaml

# currently supported biases
BIASES = ['DummyBias', 'AnchoringBias', 'LossAversion', 'HaloEffect']


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
    bias = BIASES[0] # TODO: load bias from command line arguments
    
    # Load the pre-defined YAML file for the selected cognitive bias
    # TODO: Can be entirely removed as config file loaded is encapsuled inside the TestGenerator class
    try:
        with open(f'biases/{bias}.yml') as f:
            bias_dict = yaml.safe_load(f)
    except Exception as e:
        bias_dict = {}

    # Generate a dummy test case and print it 
    model = RandomModel()
    # Load the respective test generator for bias
    generator = gen.get_generator(bias)
    test_case = generator.generate(model, bias_dict, scenario) # TODO: No need to pass bias_dict here as config is loaded internally in the generator
    print(test_case)