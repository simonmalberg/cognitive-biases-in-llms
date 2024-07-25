from utils import get_generator, get_metric
from base import PopulationError, DecisionError, MetricCalculationError
from models.OpenAI.gpt import GptThreePointFiveTurbo, GptFourO
import random


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
    bias = 'HindsightBias'
    # declare the population model
    population_model = GptFourO()
    # declare the decision model (might differ from the population model)
    decision_model = GptThreePointFiveTurbo()
    # Load the respective test generator for bias
    # Define a cognitive bias to test
    bias = "StatusQuoBias"  # TODO: optionally load bias from command line arguments
    
    # Load the test generator and metric for the bias
    generator = get_generator(bias)
    metric = get_metric(bias)

    # Instantiate the population and decision LLMs
    population_model = GptFourO()
    decision_model = GptThreePointFiveTurbo()

    # Generate a test case
    test_case = generator.generate(population_model, scenario)
    print(test_case)

    # Decide multiple times and compute the metric
    for _ in range(1):
        try:
            decision_result = decision_model.decide(test_case)
            print(decision_result)
            computed_metric = metric.compute([(test_case, decision_result)])
            print(f'Bias metric: {computed_metric}')
        except (DecisionError, MetricCalculationError, AssertionError) as e:
            print(e)