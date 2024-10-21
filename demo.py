from core.utils import get_generator, get_metric
from models.OpenAI.model import GptThreePointFiveTurbo, GptFourO
import random


# Define a cognitive bias to test (provide the name in Pascal Case, e.g., 'IllusionOfControl')
BIAS = 'Anchoring'               

# Define other execution parameters
TEMPERATURE_GENERATION = 0.7     # LLM temperature applied when generating test cases
TEMPERATURE_DECISION = 0.0       # LLM temperature applied when deciding test cases
RANDOMLY_FLIP_OPTIONS = True     # Whether answer option order will be randomly flipped in 50% of test cases
SHUFFLE_ANSWER_OPTIONS = False   # Whether answer options will be randomly shuffled for all test cases


if __name__ == "__main__":

    # Load the pre-defined scenario strings
    with open('data/scenarios.txt') as f:
        scenarios = f.readlines()

    # Randomly pick a scenario
    scenario = random.choice(scenarios)

    # Sample a random seed
    seed = random.randint(0, 1000)
    
    # Load the test generator and metric for the bias
    generator = get_generator(BIAS)
    metric = get_metric(BIAS)

    # Instantiate the generation and decision LLMs
    generation_model = GptFourO()
    decision_model = GptThreePointFiveTurbo(RANDOMLY_FLIP_OPTIONS, SHUFFLE_ANSWER_OPTIONS)

    # Generate one test case instance for the scenario
    test_cases = generator.generate_all(generation_model, [scenario], TEMPERATURE_GENERATION, seed, num_instances=1, max_retries=5)
    print(test_cases)

    # Obtain a decision result for the generated test case
    decision_results = decision_model.decide_all(test_cases, TEMPERATURE_DECISION, seed)
    print(decision_results)

    # Calculate the bias score
    metric = metric(test_results=list(zip(test_cases, decision_results)))
    computed_metric = metric.compute()
    print(f'Bias metric per each case:\n{computed_metric}')
    aggregated_metric = metric.aggregate(computed_metric)
    print(f'Aggregated bias metric: {aggregated_metric}')