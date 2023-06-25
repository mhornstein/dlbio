import sys
from model_trainer import train

if __name__ == '__main__':
    learning_rate = 0.01
    num_epochs = 10
    rna_compete_filename = sys.argv[1]

    rbns_files = sys.argv[2:]
    model, results_df = train(rbns_files=rbns_files, mode='WEIGHTED_HIGH', set_size=64, kernel_batch_normalization=True, network_batch_normalization=True, # data parameters
        kernel_sizes=[7,15], kernels_out_channel=64, pooling_size='Global', dropout_rate=0.2, hidden_layers=[32,64], # model parameters
        num_epochs=10, batch_size=64, learning_rate=0.01, l1=0, l2=0 # training parameters
        )

    print(results_df.to_string())