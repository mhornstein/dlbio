import sys
import time
import psutil
import os
import csv
import pandas as pd
from model_trainer import train

OUT_DIR = 'results'
RESULT_FILE = f'{OUT_DIR}/results.csv'
RESULTS_HEADER = ['exp_id', 'mode', 'set_size', 'kernel_batch_normalization', 'network_batch_normalization', 'kernel_sizes',
                  'kernels_out_channel', 'pooling_size', 'dropout_rate', 'hidden_layers', 'num_epochs', 'batch_size', 'learning_rate',
                  'l1', 'l2', 'max_train_acc', 'max_train_acc_epoch', 'max_train_f1', 'max_train_f1_epoch', 'time', 'cpu', 'mem']

def draw_experiment_config():
    config = {
        'mode': 'WEIGHTED_HIGH',
        'set_size': 64,
        'kernel_batch_normalization': True,
        'network_batch_normalization': True,
        'kernel_sizes': [7, 15],
        'kernels_out_channel': 64,
        'pooling_size': 'Global',
        'dropout_rate': 0.2,
        'hidden_layers': [32, 64],
        'num_epochs': 10,
        'batch_size': 64,
        'learning_rate': 0.01,
        'l1': 0,
        'l2': 0
    }
    return config


if __name__ == '__main__':
    rna_compete_filename = sys.argv[1]
    rbns_files = sys.argv[2:]

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    if not os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(RESULTS_HEADER)
        exp_id=1
    else: # we continue from existing experiments file
        df = pd.read_csv(RESULT_FILE)
        exp_id = df['exp_id'].max() + 1

    experiment_config = draw_experiment_config()
    experiment_config['rbns_files'] = rbns_files

    start_time = time.time()
    start_cpu_percent = psutil.cpu_percent()
    start_memory_usage = psutil.virtual_memory().percent

    model, results_df = train(**experiment_config)

    total_time = time.time() - start_time
    cpu_usage = psutil.cpu_percent() - start_cpu_percent
    memory_usage = psutil.virtual_memory().percent - start_memory_usage

    system_measurements = {'time': total_time, 'cpu': cpu_usage, 'mem': memory_usage}
    print(system_measurements)

