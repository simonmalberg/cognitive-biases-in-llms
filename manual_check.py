import pandas as pd
import os

def check_manually(path_to_dataset_folder: str, biases: list[str], n: int = 10, seed: int = 0) -> int:
    """
    This function samples n corresponding tests from the dataset and initiates a manual check by the user.
    
    Args:
        path_to_dataset_folder (str): the path to the dataset folder
        biases (list[str]): the biases to sample tests from
        n (int): the number of tests to sample
        seed (int): the seed for the random sampling
    
    Returns:
        int: 0 if the manual check was completed successfully
    """
    # clumsy fix to mitigate the issue of whitespaces within the bias name
    if len(biases) == 1:
        path_to_dataset = os.path.join(path_to_dataset_folder, f'{"".join(biases[0].split())}_dataset.csv')
    else:
        path_to_dataset = os.path.join(path_to_dataset_folder, 'dataset.csv')
    dataset = pd.read_csv(path_to_dataset)
    sample = dataset[dataset['bias'].isin(biases)].sample(n=n, random_state=seed)
    for bias in biases:
        print(f'Start of the review for: {bias}. Answer 1 if the test is correct, 0 if it is not.')
        for i, row in sample.iterrows():
            correctness = None
            print('-----------------------------------')
            print(f'SCENARIO:\n{row["scenario"]}\n')
            print(f'CONTROL:\n{row["control"]}')
            print(f'TREATMENT:\n{row["treatment"]}')
            while correctness not in ['1', '0']:
                correctness = input('Correct? (1 - yes/0 - no): ')  
            # add the correctness to the initial dataset
            dataset.loc[i, 'correct'] = int(correctness)
            
    if len(biases) == 1:
        dataset.to_csv(f'{path_to_dataset_folder}/checked_{"".join(biases[0].split())}_dataset.csv', index=False)
    else:
        dataset.to_csv(f'{path_to_dataset_folder}/checked_dataset.csv', index=False)
    
    return 0

if __name__ == '__main__':
    
    check_manually(path_to_dataset_folder='datasets',
                   # please, write the bias with a whitespace between words: e.g., 'Bandwagon Effect' instead of 'BandwagonEffect'
                   biases=['Anchoring'],
                   n=10,
                   seed=0)
