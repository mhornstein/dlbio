import sys
from model_trainer import train
from experimenter import model_rna_compete_predictions, log_training_results
from data_util import create_rna_seqs_tensor
import time
import numpy as np

CHOSEN_CONFIG = { # TODO change me to the final configuration you found!
    'mode': 'HIGH',
    'set_size': 100,
    'embedding_dim': 5,
    'kernel_batch_normalization': True,
    'network_batch_normalization': True,
    'kernel_size': 9,
    'stride': 1,
    'kernels_out_channel': 250,
    'pooling_size': 'Global',
    'dropout_rate': 0,
    'hidden_layers': [64, 32],
    'num_epochs': 3,
    'batch_size': 32,
    'learning_rate': 0.01,
    'l1': 0.01,
    'l2': 0.02
}

RESULT_FILE = 'result.txt'

'''
Input: 
    The 1st argument is the RNAcompete filename, and 4-6 filenames of RBNS files
Output:
    a file with RNA binding intensities (in the same order of the RNA sequences)
Flow:
    1. Parse RBNS files.
    2. create positive + negative examples.
    3. train the model.
    4. Get model classification on rna_compete_file intensities
    5. Create resulsts file.
'''

def evaluature_RBP(config, rna_seqs_tensor, rbns_files_list, result_file_path):
    train_config = config.copy()
    train_config['rbns_files'] = rbns_files_list

    print('Start training...')
    start_time = time.time()
    model, training_results_df = train(**train_config)
    total_time = time.time() - start_time
    print(f'Done training. Total time: {total_time} seconds.')

    log_training_results(results_df=training_results_df, path='training_results')  # TODO delete upon submission

    print('Start predictions...')
    predictions = model_rna_compete_predictions(model, rna_seqs_tensor)
    print(f'Done predictions. Writing results to {result_file_path} file.')
    np.savetxt(result_file_path, predictions.numpy())
    print('Done.')

if __name__ == '__main__':
    rna_compete_file = sys.argv[1]
    rbns_files_list = sys.argv[2:]

    rna_seqs_tensor = create_rna_seqs_tensor(rna_compete_file)
    evaluature_RBP(CHOSEN_CONFIG, rna_seqs_tensor, rbns_files_list, RESULT_FILE)
