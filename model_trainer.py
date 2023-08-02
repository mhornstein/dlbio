'''
This script contains the "train" function, which is used by the evaluator.py, experimenter.py, and tester.py.
It used to train a specific model based on a provided configuration.
The function returns the trained model and the training results.
All other functions in this script are exclusively used by the "train" function.
'''

from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import time
from model import ConvNet
from data_util import create_dataset
import torch
import pandas as pd

EPS = 1e-12
MAX_EXPERIMENT_TIME_IN_MINUTES = 60

def create_data_loader(X, y, batch_size, shuffle):
    '''
    creates a data loader from the given input data X and corresponding labels y.
    shuffle indicates whether to shuffle the data during batching, and batch_size used for grouping samples during training or evaluation
    '''
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def trim_single_samples_in_batch(X_train, X_val, y_train, y_val, batch_size):
    '''
    Avoid batches of size 1 as it will interfer with the model operation (mainly will damage the batch normalization functionality)
    samples that end up in batches of size 1 will be removed
    '''
    if len(X_train) % batch_size == 1:
        X_train = X_train[:-1]
        y_train = y_train[:-1]

    if len(X_val) % batch_size == 1:
        X_val = X_val[:-1]
        y_val = y_val[:-1]

    return X_train, X_val, y_train, y_val

def calc_scores(dataloader, model, loss_function, l1):
    '''
    This function calculates evaluation scores, including the average loss and accuracy, for a given model on a dataset provided through the dataloader.
    The function supports L1 regularization.
    '''
    loss_sum = 0
    good_sum = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            y_pred = model(X)
            loss = loss_function(y_pred, y)

            if l1 > 0: # Add L1 regularization
                l1_loss = 0
                for param in model.parameters():
                    l1_loss += torch.norm(param, p=1)
                loss += l1 * l1_loss

            loss_sum += loss.item()

            prediction = torch.round(y_pred)
            correct_pred = (prediction == y).sum().item()
            good_sum += correct_pred

            total += len(y)

        avg_loss = loss_sum / total
        avg_acc = good_sum / total

    return avg_loss, avg_acc

def train(
        rbns_files, mode, set_size,  # data parameters
        embedding_dim, kernel_size, stride, kernels_out_channel, pooling_size, dropout_rate, hidden_layers, kernel_batch_normalization, network_batch_normalization,  # model parameters
        num_epochs, batch_size, learning_rate, l1, l2,  # training parameters
        measure_performance # performance measuring parameters
    ):
    '''
    This function trains a CNN model on RBNS data for competing prediction.
    The training process includes data preprocessing, model construction, and optimization using backpropagation with an Adam optimizer.
    The function provides a trained model and a DataFrame with the training results for analysis.
    Parameters:
        rbns_files (list of str): A list containing file paths to RBNS data files.
        mode (str): Specifies the type of data to be used for training as positive samples. Options may be 'HIGH', 'WEIGHTED_HIGH', 'WEIGHTED_LOW', 'LOW'.
        set_size (int): The size of the positive and negative samples in the training dataset.
        embedding_dim (int or str): The configuration of the required embeddings. Can be either ONE_HOT or an integer indicating its dimension.
        kernel_size (int): The size of the kernel used in the convolutional layer for feature extraction.
        stride (int): The stride used in the convolutional layer.
        pooling_size (int or str): The size of the pooling operation. If 'Global', global max-pooling is used; otherwise, the integer value specifies the size of max-pooling.
        dropout_rate (float): The dropout rate used for regularization during training.
        hidden_layers (list of int): A list containing the dimensions of hidden layers used in the fully connected part of the model.
        kernel_batch_normalization (bool): A boolean value indicating whether batch normalization is applied to the convolutional layer.
        network_batch_normalization (bool): A boolean value indicating whether batch normalization is applied to the fully connected layers.
        num_epochs (int): The number of training epochs.
        batch_size (int): The batch size used for training.
        learning_rate (float): The learning rate used for weight updates during optimization.
        l1 (float): The strength of L1 regularization.
        l2 (float): The strength of L2 regularization.
        measure_performance (boolean): When true, train and validaion sets accuracy and loss will be evaluated and logged. When false - they will appear as None.

    Returns:
        model (torch.nn.Module): The trained CNN model
        results_df (pandas DataFrame): A DataFrame containing the training results, including training loss, training accuracy, validation loss, validation accuracy, and the time taken for each epoch.
        The DataFrame is indexed by epoch number.
    '''

    samples, labels = create_dataset(rbns_files, mode, set_size)
    input_length = samples.shape[-1]

    X_train, X_val, y_train, y_val = train_test_split(samples, labels, test_size=0.4, random_state=42)
    X_train, X_val, y_train, y_val = trim_single_samples_in_batch(X_train, X_val, y_train, y_val, batch_size)

    train_dataloader = create_data_loader(X_train, y_train, batch_size, True)
    val_dataloader = create_data_loader(X_val, y_val, batch_size, False)

    model = ConvNet(input_length=input_length,
                    embedding_dim=embedding_dim,
                    hidden_layers=hidden_layers,
                    pooling_size=pooling_size,
                    dropout_rate=dropout_rate,
                    kernel_size=kernel_size,
                    stride=stride,
                    kernels_out_channel=kernels_out_channel,
                    kernel_batch_normalization=kernel_batch_normalization,
                    network_batch_normalization=network_batch_normalization)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)

    results = []
    experiment_start_time = time.time()

    # First - add the initial results of the model (so we'll know what the base-line is)
    if measure_performance:
        model.eval()
        train_loss, train_acc = calc_scores(train_dataloader, model, loss_function, l1)
        val_loss, val_acc = calc_scores(val_dataloader, model, loss_function, l1)
    else:
        train_loss = train_acc = val_loss = val_acc = None

    result_entry = {'epoch': 0,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'epoch_time': 0}
    results.append(result_entry)

    # Then - start training
    for epoch in range(1, num_epochs + 1):
        if (time.time() - experiment_start_time > MAX_EXPERIMENT_TIME_IN_MINUTES * 60): # the experiment took more than MAX_EXPERIMENT_TIME_IN_MINUTES
            break

        epoch_start_time = time.time()

        model.train()
        for X, y in train_dataloader:
            optimizer.zero_grad()

            y_pred = model(X)
            loss = loss_function(y_pred, y)

            if l1 > 0: # Add L1 regularization
                l1_loss = 0
                for param in model.parameters():
                    l1_loss += torch.norm(param, p=1)
                loss += l1 * l1_loss

            loss.backward()
            optimizer.step()

        if measure_performance:
            model.eval()
            train_loss, train_acc = calc_scores(train_dataloader, model, loss_function, l1)
            val_loss, val_acc = calc_scores(val_dataloader, model, loss_function, l1)
        else:
            train_loss = train_acc = val_loss = val_acc = None

        epoch_time = time.time() - epoch_start_time

        result_entry = {'epoch': epoch,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'epoch_time': epoch_time}
        results.append(result_entry)

    results_df = pd.DataFrame(results).set_index('epoch')
    return model, results_df