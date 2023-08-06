'''
This script is used for training and evaluating the chosen neural networks architecture for an *individual* RBP.

Usage:
python evaluator.py [path to RNAcompete_sequences file] [one or more RBNS files for training]

example:
python evaluator.py ./data/RNAcompete_sequences.txt ./data/RBNS_training/RBP2_5nM.seq ./data/RBNS_training/RBP2_20nM.seq ./data/RBNS_training/RBP2_80nM.seq ./data/RBNS_training/RBP2_320nM.seq ./data/RBNS_training/RBP2_1300nM.seq ./data/RBNS_training/RBP2_input.seq

The script general flow is:
1. Parse RBNS files.
2. Create positive + negative samples.
3. Train the model according to the final chosen architecture.
4. Get model classification on rna_compete_file intensities.

Results:
* A model will be trained according to the RBNs files.
* The progress will be displayed in the console.
* The prediction probabilities (i.e. the scores) will be written to the generated scores.txt file.
'''

import sys
from encoding_util import ONE_HOT
from model_trainer import train
from experimenter import model_rna_compete_predictions, log_training_results
from data_util import create_rna_seqs_tensor
import time
import numpy as np

CHOSEN_CONFIG = {
    'mode': 'HIGH',
    'set_size': 1000000,
    'embedding_dim': ONE_HOT,
    'kernel_batch_normalization': False,
    'network_batch_normalization': False,
    'kernel_size': 5,
    'stride': 1,
    'kernels_out_channel': 800,
    'pooling_size': 'Global',
    'dropout_rate': 0,
    'hidden_layers': [800],
    'num_epochs': 2,
    'batch_size': 64,
    'learning_rate': 0.001,
    'l1': 0,
    'l2': 0
}

SCORES_FILE = 'scores.txt'

def evaluature_RBP(config, rna_seqs_tensor, rbns_files_list, predictions_file_path, train_result_path = None):
    train_config = config.copy()
    train_config['rbns_files'] = rbns_files_list
    train_config['measure_performance'] = True if train_result_path is not None else False

    print('Start training...')
    start_time = time.time()
    model, training_results_df = train(**train_config)
    total_time = time.time() - start_time
    print(f'Done training. Total time: {total_time} seconds. Number of epochs done: {max(training_results_df.index)}.')

    if train_result_path is not None:
        log_training_results(results_df=training_results_df, path=train_result_path)

    print('Start predictions...')
    predictions = model_rna_compete_predictions(model, rna_seqs_tensor)
    print(f'Done predictions. Writing results to {predictions_file_path} file.')
    np.savetxt(predictions_file_path, predictions.numpy())
    print('Done.')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Some arguments are missing. Please read the README file carefully.')
        sys.exit(1)

    rna_compete_file = sys.argv[1]
    rbns_files_list = sys.argv[2:]

    rna_seqs_tensor = create_rna_seqs_tensor(rna_compete_file)
    evaluature_RBP(CHOSEN_CONFIG, rna_seqs_tensor, rbns_files_list, SCORES_FILE)
