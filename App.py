from utils import get_generator, get_metric
from base import PopulationError, DecisionError, MetricCalculationError
from models.OpenAI.gpt import GptThreePointFiveTurbo, GptFourO
from models.Llama.model import LlamaThreePointOneSeventyB
import random


if __name__ == "__main__":

    # Load the pre-defined scenario strings
    with open('scenarios.txt') as f:
        scenarios = f.readlines()
    
    # Format the scenario strings by removing any markdown
    scenarios = [s.strip().replace('**', '') for s in scenarios]

    # Randomly pick a scenario
    scenario = random.choice(scenarios)

    # Define a cognitive bias to test
    bias = 'FramingEffect'
    # Define seed and temperature
    seed = random.randint(0, 1000)
    temperature = 0.7
    
    # Load the test generator and metric for the bias
    generator = get_generator(bias)
    metric = get_metric(bias)

    # Instantiate the population and decision LLMs
    population_model = GptFourO()
    decision_model = LlamaThreePointOneSeventyB(shuffle_answer_options=True)
    
    # Generate test cases and decide for all given scenarios and compute the metric
    try:
        test_cases = generator.generate_all(population_model, [scenario], temperature, seed, num_instances=1, max_retries=5)
        print(test_cases)
        decision_results = decision_model.decide_all(test_cases, temperature, seed)
        print(decision_results)
        computed_metric = metric.compute(list(zip(test_cases, decision_results)))
        print(f'Bias metric: {computed_metric}')
    except (DecisionError, MetricCalculationError, AssertionError) as e:
        print(e)