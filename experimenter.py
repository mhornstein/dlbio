import sys
import time
import psutil
import os
import csv
import pandas as pd
import torch
from matplotlib import pyplot as plt
import random
import scipy.stats as stats
from model_trainer import train
from data_util import create_rna_seqs_tensor, load_intensities_file
from scipy.stats import pearsonr
from rbns_files_list import rbns_files_list
from encoding_util import ONE_HOT

EXPERIMENT_COUNT = 100000
OUT_DIR = 'results'
MEASUREMENTS_FILE = f'{OUT_DIR}/measurements.csv'
MEASUREMENTS_HEADER =   ['exp_id',
                        # data parameters
                        'protein_index', 'mode', 'set_size',
                        # model parameters
                        'embedding_dim', 'kernel_batch_normalization', 'network_batch_normalization', 'kernel_size', 'stride', 'kernels_out_channel', 'pooling_size', 'dropout_rate', 'hidden_layers',
                        # training parameters
                        'num_epochs', 'batch_size', 'learning_rate', 'l1', 'l2',
                        # experiment measurements
                        'max_train_acc', 'max_train_acc_epoch',
                        'max_val_acc', 'max_val_acc_epoch',
                        # system measurements
                        'time', 'cpu', 'mem',
                        # pearson correlation
                       'pearson correlation']

def draw_experiment_config():
    protein_index = random.randint(0, 14)
    mode = 'HIGH' # random.choice(['WEIGHTED_HIGH', 'WEIGHTED_LOW', 'HIGH', 'LOW'])
    set_size = 10000
    embedding_dim = random.choice([ONE_HOT, 3, 5, 10, 12]) # Use None for one-hot embeddings.
    kernel_batch_normalization = random.choice([True, False])
    network_batch_normalization = random.choice([True, False])
    kernel_size = random.choice([5, 7, 9, 11, 15])
    stride = random.choice([1, 2, 3, 4, 5])
    kernels_out_channel = random.choice([32, 64, 128, 256, 512])
    pooling_size = 'Global' if kernels_out_channel >= 256 else random.choice(['Global', 2, 3])
    dropout_rate = random.choice([0, 0.25, 0.5])
    hidden_layers = random.choices([32, 64, 128], k=random.randint(1, 3))
    num_epochs = 200
    batch_size = random.choice([256, 128, 64])
    learning_rate = stats.loguniform.rvs(0.0005, 0.05)
    l1 = random.choice([0.1, 0.001, 0.0001, 0])
    l2 = random.choice([0.1, 0.001, 0.0001, 0])

    config = {
        'protein_index': protein_index,
        'mode': mode,
        'set_size': set_size,
        'embedding_dim': embedding_dim,
        'kernel_batch_normalization': kernel_batch_normalization,
        'network_batch_normalization': network_batch_normalization,
        'kernel_size': kernel_size,
        'stride': stride,
        'kernels_out_channel': kernels_out_channel,
        'pooling_size': pooling_size,
        'dropout_rate': dropout_rate,
        'hidden_layers': hidden_layers,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'l1': l1,
        'l2': l2
    }
    return config

def calc_experiment_measurements(results_df):
    measurements = {}
    max_train_acc = results_df['train_acc'].max()
    max_train_acc_epoch = results_df['train_acc'].idxmax()
    measurements['max_train_acc'] = max_train_acc
    measurements['max_train_acc_epoch'] = max_train_acc_epoch

    max_val_acc = results_df['val_acc'].max()
    max_val_acc_epoch = results_df['val_acc'].idxmax()
    measurements['max_val_acc'] = max_val_acc
    measurements['max_val_acc_epoch'] = max_val_acc_epoch

    return measurements

def create_measurement_entry(exp_id, experiment_config, experiment_measurements, system_measurements, pearson_corr):
    entry = {'exp_id': exp_id}
    entry.update(experiment_config)
    entry.update(experiment_measurements)
    entry.update(system_measurements)
    entry['pearson correlation'] = pearson_corr
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

def model_rna_compete_predictions(model, rna_seqs_tensor):
    model.eval()
    with torch.no_grad():
        predictions = model(rna_seqs_tensor)
    return predictions

def to_train_config(experiment_config, rbns_files_list):
    '''
    For train configuration we need to remove the protein index and replace it with the list of its corresponding files
    '''
    train_config = experiment_config.copy()
    protein_index = train_config['protein_index']
    del train_config['protein_index']
    train_config['rbns_files'] = rbns_files_list[protein_index][1:]
    return train_config

if __name__ == '__main__':
    # Read and convert RNA sequences to tensor
    rna_compete_file = "./data/RNAcompete_sequences.txt"
    rna_seqs_tensor = create_rna_seqs_tensor(rna_compete_file)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    if not os.path.exists(MEASUREMENTS_FILE):
        with open(MEASUREMENTS_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(MEASUREMENTS_HEADER)
        start_exp_id = 1
    else: # we continue from existing experiments file
        df = pd.read_csv(MEASUREMENTS_FILE)
        start_exp_id = 1 if len(df['exp_id']) == 0 else df['exp_id'].max() + 1

    for exp_id in range(start_exp_id, start_exp_id + EXPERIMENT_COUNT):
        experiment_config = draw_experiment_config()
        experiment_config_str = ','.join([f'{key}={value}' for key, value in experiment_config.items()])
        print(f'Running experiment {exp_id}:', experiment_config_str)

        start_time = time.time()
        start_cpu_percent = psutil.cpu_percent()
        start_memory_usage = psutil.virtual_memory().percent

        train_config = to_train_config(experiment_config, rbns_files_list)
        model, experiment_results_df = train(**train_config)

        total_time = time.time() - start_time
        cpu_usage = psutil.cpu_percent() - start_cpu_percent
        memory_usage = psutil.virtual_memory().percent - start_memory_usage

        log_experiment_results(OUT_DIR, exp_id, experiment_results_df)

        predictions = model_rna_compete_predictions(model, rna_seqs_tensor)

        # Compare model predictions to intensities file by Pearson Correlation
        protein_index = experiment_config['protein_index']
        intensities = load_intensities_file(rbns_files_list[protein_index][0])
        corr, _ = pearsonr(predictions.numpy().flatten(), intensities)

        system_measurements = {'time': total_time, 'cpu': cpu_usage, 'mem': memory_usage}
        experiment_measurements = calc_experiment_measurements(experiment_results_df)
        entry = create_measurement_entry(exp_id, experiment_config, experiment_measurements, system_measurements, corr)

        write_measurement(MEASUREMENTS_FILE, MEASUREMENTS_HEADER, entry)