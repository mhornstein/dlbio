import os
from data_util import load_intensities_file
from scipy.stats import pearsonr

def get_rbns_name(filename):
    end_index = filename.index(".")
    return filename[:end_index]

def get_rbns_number(filename):
    start_index = filename.index("RBP") + 3  # Adding 3 to skip "RBP"
    end_index = filename.index(".")
    number = int(filename[start_index:end_index])
    return number

if __name__ == '__main__':
    train_gold_dir = 'data/RNCMPT_training'
    train_predictions_dir = 'train_set_results'

    files_list = os.listdir(train_gold_dir)

    for file in files_list:
        rbns_number = get_rbns_number(file)
        rbns_name = get_rbns_name(file)
        gold_intensities = load_intensities_file(f'{train_gold_dir}/{file}')
        predicted_intensities = load_intensities_file(f'{train_predictions_dir}/{file}')
        corr, _ = pearsonr(gold_intensities, predicted_intensities)
        print(rbns_number, rbns_name, corr)
    print()

