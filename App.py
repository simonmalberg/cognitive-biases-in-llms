from utils import get_generator, get_metric
from models import RandomModel, GptThreePointFiveTurbo, PopulationError, DecisionError, GptFourO
from base import MetricCalculationError
import random


# currently supported biases
BIASES = ['AnchoringBias', 'LossAversion', 'ConfirmationBias']  # TODO: Halo Effect test needs to be fixed


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
    bias = 'OptimismBias'
    # declare the population model
    population_model = GptThreePointFiveTurbo()
    # declare the decision model (might differ from the population model)
    decision_model = GptThreePointFiveTurbo()
    # Load the respective test generator for bias
    generator = get_generator(bias)
    # Load the respective metric for the bias
    metric = get_metric(bias)
    try:
        test_case = generator.generate(population_model, scenario)
        print(test_case)
        decision_result = decision_model.decide(test_case)
        print(decision_result)
        computed_metric = metric.compute([(test_case, decision_result)])
        print(f'Bias metric: {computed_metric}')
    except (PopulationError, DecisionError, MetricCalculationError, AssertionError) as e:
        print(e)
        print("Test case is failed. Exiting...")
        exit(1)
