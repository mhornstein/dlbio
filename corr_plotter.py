import os
from data_util import load_intensities_file
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt

def get_rbns_number(filename):
    start_index = filename.index("RBP") + 3  # Adding 3 to skip "RBP"
    end_index = filename.index(".")
    number = int(filename[start_index:end_index])
    return number

'''
This script is used to calculate the correlation between the gold and predicted scores of the RNCMPT training data.
The script requires the following inputs:

train_gold_dir: A string representing the directory path where the gold (actual) scores for the RNCMPT training data are stored.
It should contain files of the form RBP1.txt...RBP16.txt.

train_predictions_dir: A string representing the directory path where the predicted scores for the RNCMPT training data are stored. 
These predicted scores are the output of the model's performance on the training data. They can be created using the tester.py script.
It should contain files of the same form as train_gold_dir, i.e. RBP1.txt...RBP16.txt.

results_dir: The directory where the results should be saved.
The result will contain a CSV containing the correlation scores and a corresponding box plot.
'''
if __name__ == '__main__':
    train_gold_dir = 'data/RNCMPT_training'
    train_predictions_dir = 'train_set_results'
    results_dir = 'corr_report'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    files_list = os.listdir(train_gold_dir)

    # create a df with the correlation calculation results
    data = [(get_rbns_number(file),
             pearsonr(load_intensities_file(f'{train_gold_dir}/{file}'),
                      load_intensities_file(f'{train_predictions_dir}/{file}'))[0])
            for file in files_list]
    df = pd.DataFrame(data, columns=['rbns_number', 'corr'])
    df.set_index('rbns_number', inplace=True)  # Set 'rbns_number' as the index
    df.sort_index(inplace=True)

    df.to_csv(f'{results_dir}/correlation_data.csv')

    plt.boxplot(df['corr'])
    plt.title('Correlation Box Plot')
    plt.ylabel('Correlation')
    plt.savefig(f'{results_dir}/correlation_box_plot.png')
    plt.close()

