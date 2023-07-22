import sys
import os
from evaluator import CHOSEN_CONFIG, evaluature_RBP
from data_util import get_RBNS_files_for_protein, create_rna_seqs_tensor
import time

def extract_RBP_number(rbp_filename):
    start_index = rbp_filename.index("RBP") + 3  # Adding 3 to skip "RBP"
    end_index = rbp_filename.index("_")
    number = int(rbp_filename[start_index:end_index])
    return number

def get_rbps_indexes_from_dir(rbns_testing_dir):
    files_list = os.listdir(rbns_testing_dir)
    indexes = set(map(extract_RBP_number, files_list))
    return indexes

'''
Input:
    The 1st argument is the RNAcompete filename, and the second is the directory of the RBNS-test files.
    The 3rd argument is the path for the required result directory (can be either relative or absolute path).
Output:
    The result directory will be created. The directory will contain a file with RNA binding intensities
    (in the same order of the RNA sequences) for each RBP founds in the RBNS-test folder.
    e.g. RBP19.txt will contain the intensities for RBP 19.
'''
if __name__ == '__main__':
    start_time = time.time()
    rna_compete_file = sys.argv[1]
    rbns_testing_dir = sys.argv[2]
    results_dir = sys.argv[3]
    train_result_path = f'{results_dir}/train_result'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if not os.path.exists(train_result_path):
        os.makedirs(train_result_path)

    rna_seqs_tensor = create_rna_seqs_tensor(rna_compete_file)

    indexes = get_rbps_indexes_from_dir(rbns_testing_dir)
    for protein_index in indexes:
        print(f'Testing protein {protein_index}')
        rbns_files_list = get_RBNS_files_for_protein(rbns_testing_dir, protein_index)
        predictions_file_path = f'{results_dir}\\RBP{protein_index}.txt'
        evaluature_RBP(config=CHOSEN_CONFIG, rna_seqs_tensor=rna_seqs_tensor, rbns_files_list=rbns_files_list,
                       predictions_file_path=predictions_file_path, train_result_path=f'{train_result_path}/{protein_index}')
        print()

    print(f'Done. Total time: {time.time()-start_time} seconds.')