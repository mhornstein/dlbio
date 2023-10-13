This repository contains 4 scripts.

The main script is `evaluator.py`, which operates according to the assignment requirements. Further explanations about its operation are provided below. Note that it uses code imported from the other code files submitted, so please make sure they appear in the same directory upon execution.

However, we also provide all the code we made use of as part of the content of this project, along with instructions on how to run each.

Please refer to the "Report.pdf" file for comprehensive explanations and details of all the experiments and their results.

# Scripts and Their Purpose

1. `evaluator.py`: used for training and evaluating the chosen neural networks architecture for an *individual* RBP.
2. `experimenter.py`: This script automates the process of conducting experiments with various neural network configurations. It runs a given number of experiments, in each of which a random configuration and an RBP are sampled out of the training-RBPs (i.e., RBP 1-16). The training and evaluation results of each experiment are recorded.
3. `tester.py`: used for training and conducting predictions using the chosen neural network configuration for a set of RBPs.
4. `corr_plotter.py`: a small utility script for calculating and visualizing the correlations between RBPs' gold intensities and the predictions.

These scripts all use PyTorch, so PyTorch warnings might be presented in the console as well as the scripts' output.

# Instructions for Running Each Script

## `evaluator.py`

Usage:
```bash
python evaluator.py [path to RNAcompete_sequences file] [one or more RBNS files for training]
```

Example:
```bash
python evaluator.py ./data/RNAcompete_sequences.txt ./data/RBNS_training/RBP2_5nM.seq ./data/RBNS_training/RBP2_20nM.seq ./data/RBNS_training/RBP2_80nM.seq ./data/RBNS_training/RBP2_320nM.seq ./data/RBNS_training/RBP2_1300nM.seq ./data/RBNS_training/RBP2_input.seq
```

The script's general flow is as follows:
1. Parse RBNS files.
2. Create positive and negative samples.
3. Train the model according to the final chosen architecture.
4. Get the model's classification on `rna_compete_file` intensities.

Results:
- A model will be trained according to the RBNs files.
- The progress will be displayed in the console.
- The prediction probabilities (i.e., the scores) will be written to the generated `scores.txt` file.

## `experimenter.py`

Usage:
```bash
python experimenter.py [path to RNAcompete_sequences file] [path to RBNS_training directory] [path to RNCMPT_training directory] [number of experiments to conduct]
```

Example:
```bash
python experimenter.py ./data/RNAcompete_sequences.txt ./data/RBNS_training ./data/RNCMPT_training 5
```

Results:
- The progress will be displayed in the console.
- A "result" directory will be created (if not already existed). It will contain:
  - `measurements.csv`: a CSV file logging each experiment ID, the sampled configuration, and the experiment results (loss, accuracy, time, etc).
  - A directory for each experiment ID, containing accuracy and loss graphs, as well as the raw training results data.

Note: If you wish to conduct more experiments, simply re-run `experimenter.py` again. It will fetch the last experiment ID from `measurements.csv` and continue from there.

## `tester.py`

Usage:
```bash
python tester.py [path to RNAcompete_sequences file] [directory of the RBNS-test files] [result directory]
```

Example:
```bash
python tester.py ./data/RNAcompete_sequences.txt ./data/RBNS_training train_results
```

or:

```bash
python tester.py ./data/RNAcompete_sequences.txt ./data/RBNS_testing test_results
```

Results:
- A model will be trained for each RBP found in the directory.
- The trained model will then be used to predict the intensity of the RNA sequences provided in the RNAcompete sequences file:
	* The prediction probabilities will be written to separate text files, with each file named after the corresponding RBP, e.g. RBP1.txt will contain the predictions for RBP number 1.
	* The directory will also include a "train_result" directory with individual directories for each RBP, containing information about the training performance.
- The progress will be displayed in the console.

## `corr_plotter.py`

Usage:
```bash
python corr_plotter.py [path to gold-scores directory] [path to predicted-scores directory] [results directory]
```

Example:
```bash
python corr_plotter.py data/RNCMPT_training train_set_results corr_report
```
where:
data/RNCMPT_training – was provided as part of the assignment.
train_set_results – is the output directory of the tester.py script.

Explanation of parameters:

- `path to gold-scores directory`: A string representing the directory path where the gold (actual) scores for the RNCMPT training data are stored. It should contain files of the form `RBP1.txt`, `RBP2.txt`, etc.
- `path to predicted-scores directory`: A string representing the directory path where the predicted scores for the RNCMPT training data are stored. These predicted scores are the output of the model's predictions for the training data and can be created using the `tester.py` script. It should contain files of the same form as gold_dir, i.e. RBP1.txt...RBP16.txt.
- `results directory`: The directory where the results should be saved.

Results:
- The result will contain a CSV containing the correlation scores and a corresponding box plot.

# Requirements

The necessary libraries that need to be installed are listed in the `requirements.txt` file. You can install all of them using `pip install`.

Note: We use the slim version of PyTorch as we ran the scripts only on machines that do not have GPUs.

## Tests

All code was rigorously tested on:

- Windows machine with Python 3.10.8.
- Unix machine with Python 3.9.2.
