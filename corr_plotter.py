'''
Small utility script for calculating and visualizing the correlations between RBPs' gold intensities and the predictions.

Usage:
python corr_plotter.py [path to gold-scores directory] [path to predicted-scores directory] [results directory]

example:
python corr_plotter.py data/RNCMPT_training train_set_results corr_report
where:
data/RNCMPT_training – was provided as part of the assignment.
train_set_results – is the output directory of the tester.py script.

Explainations about the parameters:
* path to gold-scores directory: A string representing the directory path where the gold (actual) scores for the RNCMPT training data are stored.
It should contain files of the form RBP1.txt...RBP16.txt.
* path to predicted-scores directory: A string representing the directory path where the predicted scores for the RNCMPT training data are stored.
These predicted scores are the output of the model's predictions for the training data. They can be created using the tester.py script.
It should contain files of the same form as gold_dir, i.e. RBP1.txt...RBP16.txt.
* results directory: The directory where the results should be saved.

Results:
The result will contain a CSV containing the correlation scores and a corresponding box plot.
'''

import os
from data_util import load_intensities_file
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import sys

def get_rbns_number(filename):
    '''
    Extracts RBP number from filename
    '''
    start_index = filename.index("RBP") + 3  # Adding 3 to skip "RBP"
    end_index = filename.index(".")
    number = int(filename[start_index:end_index])
    return number

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Some arguments are missing. Please read the README file carefully.')
        sys.exit(1)

    train_gold_dir = sys.argv[1]
    train_predictions_dir = sys.argv[2]
    results_dir = sys.argv[3]

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

