import sys
import time
import psutil
import os
import csv
import pandas as pd
from matplotlib import pyplot as plt

from model_trainer import train

OUT_DIR = 'results'
MEASUREMENTS_FILE = f'{OUT_DIR}/measurements.csv'
MEASUREMENTS_HEADER =   ['exp_id',
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

def create_measurement_entry(exp_id, experiment_config, experiment_measurements, system_measurements):
    entry = {'exp_id': exp_id}
    entry.update(experiment_config)
    entry.update(experiment_measurements)
    entry.update(system_measurements)
    del entry['rbns_files']
    return entry

def write_measurement(measurement_file, measurement_header, entry):
    esc_value = lambda val: str(val).replace(',', '') # remove commas from values' conent, so csv format won't be damaged
    with open(measurement_file, 'a') as file:
        values = [esc_value(entry[key]) for key in measurement_header]
        file.write(','.join(str(value) for value in values) + '\n')

def plot(epochs, train_data, val_data, train_label, val_label, measurement_title, file_path):
    plt.plot(epochs, train_data, label=train_label)
    plt.plot(epochs, val_data, label=val_label)

    plt.xlabel('Epoch')
    plt.ylabel(measurement_title)
    plt.title(f'{measurement_title} over Epochs')

    plt.legend()

    plt.savefig(file_path)
    plt.clf()
    plt.close()

def log_experiment_results(out_dir, exp_id, results_df):
    path = f'{out_dir}\{exp_id}'
    if not os.path.exists(path):
        os.makedirs(path)

    epochs = results_df.index

    plot(epochs=epochs, train_data=results_df['train_acc'], val_data=results_df['val_acc'],
         train_label='train accuracy', val_label='validation accuracy', measurement_title='Accuracy',
         file_path=f'{path}/accuracy.png')
    plot(epochs=epochs, train_data=results_df['train_loss'], val_data=results_df['val_loss'],
         train_label='train loss', val_label='validation loss', measurement_title='Loss',
         file_path=f'{path}/loss.png')
    results_df.to_csv(f'{path}/experiment_results.csv', index=True)


if __name__ == '__main__':
    rbns_files = sys.argv[1:]

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    if not os.path.exists(MEASUREMENTS_FILE):
        with open(MEASUREMENTS_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(MEASUREMENTS_HEADER)
        exp_id = 1
    else: # we continue from existing experiments file
        df = pd.read_csv(MEASUREMENTS_FILE)
        exp_id = df['exp_id'].max() + 1

    experiment_config = draw_experiment_config()
    experiment_config['rbns_files'] = rbns_files

    start_time = time.time()
    start_cpu_percent = psutil.cpu_percent()
    start_memory_usage = psutil.virtual_memory().percent

    model, experiment_results_df = train(**experiment_config)

    total_time = time.time() - start_time
    cpu_usage = psutil.cpu_percent() - start_cpu_percent
    memory_usage = psutil.virtual_memory().percent - start_memory_usage

    log_experiment_results(OUT_DIR, exp_id, experiment_results_df)

    system_measurements = {'time': total_time, 'cpu': cpu_usage, 'mem': memory_usage}
    experiment_measurements = calc_experiment_measurements(experiment_results_df)
    entry = create_measurement_entry(exp_id, experiment_config, experiment_measurements, system_measurements)

    write_measurement(MEASUREMENTS_FILE, MEASUREMENTS_HEADER, entry)

