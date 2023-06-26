import sys
from model_trainer import train

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

    experiment_config = draw_experiment_config()

    experiment_config['rbns_files'] = rbns_files
    model, results_df = train(**experiment_config)

    print(results_df.to_string())