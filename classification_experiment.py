import sys
import time
import psutil
import os
import csv
import pandas as pd
from model_trainer import train

OUT_DIR = 'results'
RESULT_FILE = f'{OUT_DIR}/results.csv'
RESULTS_HEADER = ['exp_id',
                  # data parameters
                  'mode', 'set_size',
                  # model parameters
                  'kernel_batch_normalization', 'network_batch_normalization', 'kernel_sizes', 'kernels_out_channel', 'pooling_size', 'dropout_rate', 'hidden_layers',
                  # training parameters
                  'num_epochs', 'batch_size', 'learning_rate', 'l1', 'l2',
                  # experiment measurements
                  'max_train_acc', 'max_train_acc_epoch', 'max_train_f1', 'max_train_f1_epoch',
                  'max_val_acc', 'max_val_acc_epoch', 'max_val_f1', 'max_val_f1_epoch',
                  # system measurements
                  'time', 'cpu', 'mem']

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

def calc_experiment_measurements(results_df):
    measurements = {}
    max_train_acc = results_df['train_acc'].max()
    max_train_acc_epoch = results_df['train_acc'].idxmax()
    measurements['max_train_acc'] = max_train_acc
    measurements['max_train_acc_epoch'] = max_train_acc_epoch

    max_train_f1 = results_df['train_f1'].max()
    max_train_f1_epoch = results_df['train_f1'].idxmax()
    measurements['max_train_f1'] = max_train_f1
    measurements['max_train_f1_epoch'] = max_train_f1_epoch

    max_val_acc = results_df['val_acc'].max()
    max_val_acc_epoch = results_df['val_acc'].idxmax()
    measurements['max_val_acc'] = max_val_acc
    measurements['max_val_acc_epoch'] = max_val_acc_epoch

    max_val_f1 = results_df['val_f1'].max()
    max_val_f1_epoch = results_df['val_f1'].idxmax()
    measurements['max_val_f1'] = max_val_f1
    measurements['max_val_f1_epoch'] = max_val_f1_epoch

    return measurements

def create_result_entry(exp_id, experiment_config, experiment_measurements, system_measurements):
    results_entry = {'exp_id': exp_id}
    results_entry.update(experiment_config)
    results_entry.update(experiment_measurements)
    results_entry.update(system_measurements)
    del results_entry['rbns_files']
    return results_entry

def write_results(result_file, result_header, result_entry):
    esc_value = lambda val: str(val).replace(',', '')
    with open(result_file, 'a') as file:
        values = [esc_value(result_entry[key]) for key in result_header]
        file.write(','.join(str(value) for value in values) + '\n')


if __name__ == '__main__':
    rna_compete_filename = sys.argv[1]
    rbns_files = sys.argv[2:]

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    if not os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(RESULTS_HEADER)
        exp_id = 1
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

    experiment_measurements = calc_experiment_measurements(results_df)

    results_entry = create_result_entry(exp_id, experiment_config, experiment_measurements, system_measurements)

    write_results(RESULT_FILE, RESULTS_HEADER, results_entry)

