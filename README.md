# **A Comprehensive Evaluation of Cognitive Biases in LLMs**  

This repository contains all code used in the research paper **"A Comprehensive Evaluation of Cognitive Biases in LLMs"**, published on **arXiv**. The paper is available [here](https://arxiv.org/abs/2410.15413). The code is made available to support researchers and practitioners interested in exploring and building on our work.

## **Table of Contents**

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Reproducing Experiments](#reproducing-experiments)
  - [Adding Tests](#adding-tests)
- [Folder Structure](#folder-structure)
- [License](#license)
- [Citation](#citation)

## **Introduction**

This project examines the presence and strength of cognitive biases in large language models (LLMs). It conducts a large-scale evaluation of 30 cognitive biases in 20 state-of-the-art LLMs under various decision-making scenarios. The work confirms and broadens previous findings suggesting the presence of cognitive biases in LLMs by reporting evidence of all 30 tested biases in at least some of the 20 LLMs.

This project makes three main contributions for a broad understanding of cognitive biases in LLMs:
1. **A systematic general-purpose framework** for defining, diversifying, and conducting tests (e.g., for cognitive biases) with LLMs.
2. **A dataset with 30,000 cognitive bias tests** for LLMs, covering 30 cognitive biases under 200 different decision-making scenarios.
3. **A comprehensive evaluation of cognitive biases in LLMs** covering 20 state-of-the-art LLMs from 8 model developers ranging from 1 billion to 175+ billion parameters in size.

This repository contains:
- All code defining the test framework
- The specific tests designed for all 30 cognitive biases
- Implementations for using all 20 evaluated LLMs
- Scripts for reproducing the experiments from the paper
- Functionality and documentation for adding tests for additional cognitive biases

The full dataset with 30,000 cognitive bias tests is not included in this repository due to its size. We are working on releasing the full dataset on HuggingFace very soon.

## **Installation**

1. **Clone the repository:**

    ```bash
    git clone https://github.com/simonmalberg/cognitive-biases-in-llms.git
    cd cognitive-biases-in-llms
    ```

2. **Set up a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    source .venv/bin/activate   # On Windows: `.venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Store API keys in system environment variables**

## **Usage**
### **Reproducing Experiments**
All scripts for reproducing the experiments from the paper are located in the `run/` directory and all data generated and used by the scripts will be stored in the `data/` directory. The scripts should be executed in the following order:

1. **Generate decision-making scenarios with the `run/scenario_generation.py` script.**

2. **Generate test case instances with the `run/test_generation.py` script.**

    ```bash
    python run/test_generation.py
    ```

    This will generate five test case instances per scenario for all biases defined in the repository. If you only want to generate fewer test case instances per scenario or only for a subset of biases, you can specify them as follows (example):

    ```bash
    python run/test_generation.py --bias "FramingEffect, IllusionOfControl" --num_instances 2
    ```

    Generated test case instances will be stored in `data/generated_tests/{bias}/` as XML and `data/generation_logs/{bias}/` as TXT files (one file per scenario with all instances generated for that scenario) or in CSV format in `data/generated_datasets/` with one CSV file per bias.

3. **(Optional) Manually check a subset of the generated test case instances with the `run/test_check.py` script.**

    ```bash
    python run/test_check.py
    ```

    This will sample ten generated test case instances per bias. If you want to conduct the check with a different sample size or for only a subset of biases, you can specify them as follows (example):

    ```bash
    python run/test_check.py --bias "FramingEffect, IllusionOfControl" --n_sample 2
    ```
    
    The script will guide the user through all sampled test case instances and ask whether the instances are correct. It will store the check results inside the `data/checked_datasets/` directory with one CSV file per bias. The files will be identical to those stored in `data/generated_datasets/` except that they have one additional column `correct` that contains `1`/`0` for all instances marked as correct/incorrect. 

4. **Obtain decision results for the generated test case instances with the `run/test_decision.py` script.**

These scripts are currently undergoing updates to point to the right directories inside the `data/` folder and take parameters from the command line. They may temporarily not work as intended.

### **Adding Tests**
To add a new cognitive bias test, run the `core/add_test.py` script. The script will prompt you to enter the name of the new cognitive bias. Enter the name using alphabetic characters only. The script will then create a new subfolder with the name of the new bias in the `tests/` directory and will automatically create three files:

- `__init__.py` is empty and does not require any further work.

- `config.xml` specifies the test templates and any custom values to sample from.

- `test.py` defines the specific `TestGenerator` and `Metric` classes implementing the generation of instances of the test case and exact metric parameterization to be used for measuring the test score.

Adjust the contents of the `config.xml` and `test.py` files to implement your test logic.

## **Folder Structure**
```bash
├── core/              # Core components of the test framework
├── data/              # Data generated or used by scripts
├── models/            # Interfaces to LLMs from different developers
├── run/               # Scripts to reproduce the experiments from the paper
├── test/              # Test case definitions
├── demo.py            # A simple demo script that generates a test case and obtains a decision result
├── LICENSE.txt        # The license for this code
├── README.md          # This file
├── requirements.txt   # List of dependencies
```

## **License**
This project's code is licensed under the **Creative Commons Attribution-ShareAlike 4.0 International Public License** (CC BY-SA 4.0). See the `LICENSE.txt` file for details.

## **Citation**
If you use this code for your research, please cite our paper:

```
@misc{malberg2024comprehensiveevaluationcognitivebiases,
    title={A Comprehensive Evaluation of Cognitive Biases in LLMs}, 
    author={Simon Malberg and Roman Poletukhin and Carolin M. Schuster and Georg Groh},
    year={2024},
    eprint={2410.15413},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2410.15413}, 
}
```