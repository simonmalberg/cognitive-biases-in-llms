# **Data Directory**

This `data/` directory holds all data generated or used by scripts. In particular, these are:

   ```bash
   ├── generated_tests/      # Raw generated test cases before they are compiled into datasets, one folder per bias with one .xml file per scenario
   ├── generation_logs/      # Readable log files for all generated test cases, one folder per bias with one .txt file per scenario
   ├── generated_datasets/   # Datasets with generated test cases, one .csv file per bias
   ├── decision_results/     # Decision results for generated test cases obtained from decision models, one .csv file per model
   ├── scenarios.txt         # A .txt file where each line represents a decision-making scenario
   ├── full_dataset.csv      # The full dataset with all generated test cases of all biases in .csv format
   ├── README.md             # This file
   ```

Due to the size of the data, the data that was generated for the paper is not pushed to this repository.