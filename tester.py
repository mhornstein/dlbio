'''
This script is used for training and conducting predictions using the chosen neural network configuration for a set of RBPs.

Usage:
python tester.py [path to RNAcompete_sequences file] [directory of the RBNS-test files] [result directory]

example:
python tester.py ./data/RNAcompete_sequences.txt ./data/RBNS_training train_results
or:
python tester.py ./data/RNAcompete_sequences.txt ./data/RBNS_testing test_results

Results:
* A model will be trained for each RBP found in the directory.
* The trained model will then be used to predict the intensity of the RNA sequences provided in the RNAcompete sequences file:
    - The prediction probabilities will be written to separate text files, with each file named after the corresponding RBP, e.g. RBP1.txt will contain the predictions for RBP number 1.
    - The directory will also include a " train_result" directory with individual directories for each RBP, containing information about the training performance.
* The progress will be displayed in the console.
'''

import sys
import os
from evaluator import CHOSEN_CONFIG, evaluature_RBP
from data_util import get_RBNS_files_for_protein, create_rna_seqs_tensor
import time

def extract_RBP_number(rbp_filename):
    '''
    Extracts the RBP number from a given RBP filename
    '''
    start_index = rbp_filename.index("RBP") + 3  # Adding 3 to skip "RBP"
    end_index = rbp_filename.index("_")
    number = int(rbp_filename[start_index:end_index])
    return number

def get_rbps_indexes_from_dir(rbns_testing_dir):
    '''
    Returns a list of RBP numbers extracted from the filenames in the given directory
    '''
    files_list = os.listdir(rbns_testing_dir)
    indexes = set(map(extract_RBP_number, files_list))
    return indexes

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Some arguments are missing. Please read the README file carefully.')
        sys.exit(1)

    rna_compete_file = sys.argv[1]
    rbns_testing_dir = sys.argv[2]
    results_dir = sys.argv[3]
    train_result_path = f'{results_dir}/train_result'

    start_time = time.time()
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if not os.path.exists(train_result_path):
        os.makedirs(train_result_path)

    rna_seqs_tensor = create_rna_seqs_tensor(rna_compete_file)

    indexes = get_rbps_indexes_from_dir(rbns_testing_dir)
    for protein_index in indexes:
        print(f'Testing protein {protein_index}')
        rbns_files_list = get_RBNS_files_for_protein(rbns_testing_dir, protein_index)
        predictions_file_path = f'{results_dir}/RBP{protein_index}.txt'
        evaluature_RBP(config=CHOSEN_CONFIG, rna_seqs_tensor=rna_seqs_tensor, rbns_files_list=rbns_files_list,
                       predictions_file_path=predictions_file_path, train_result_path=f'{train_result_path}/{protein_index}')
        print()

    print(f'Done. Total time: {time.time()-start_time} seconds.')