import generation as gen
from models import RandomModel, GptThreePointFiveTurbo
import random
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# currently supported biases
BIASES = ['AnchoringBias', 'LossAversion', 'ConfirmationBias'] #TODO: Halo Effect test needs to be fixed


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
    population_model = GptThreePointFiveTurbo()
    # Load the respective test generator for bias
    generator = gen.get_generator(bias)
    test_case = generator.generate(population_model, scenario)
    print(test_case)
    # declare the decision model (might differ from the population model)
    decision_model = GptThreePointFiveTurbo()
    decision = decision_model.decide(test_case)
    print(decision)