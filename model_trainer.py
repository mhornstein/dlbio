from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import time
from model import ConvNet
from data_util import create_dataset
import torch
import pandas as pd

EPS = 1e-12
MAX_EXPERIMENT_TIME_IN_MINUTES = 45

def create_data_loader(X, y, batch_size, shuffle):
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


def train(
        rbns_files, mode, set_size,  # data parameters
        kernel_sizes, stride, kernels_out_channel, pooling_size, dropout_rate, hidden_layers, kernel_batch_normalization, network_batch_normalization,  # model parameters
        num_epochs, batch_size, learning_rate, l1, l2  # training parameters
    ):
    samples, labels = create_dataset(rbns_files, mode, set_size)
    input_length = samples.shape[-1]

    X_train, X_val, y_train, y_val = train_test_split(samples, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = trim_single_samples_in_batch(X_train, X_val, y_train, y_val, batch_size)

    train_dataloader = create_data_loader(X_train, y_train, batch_size, True)
    val_dataloader = create_data_loader(X_val, y_val, batch_size, False)

    model = ConvNet(input_length=input_length,
                    hidden_layers=hidden_layers,
                    pooling_size=pooling_size,
                    dropout_rate=dropout_rate,
                    kernel_sizes=kernel_sizes,
                    stride=stride,
                    kernels_out_channel=kernels_out_channel,
                    kernel_batch_normalization=kernel_batch_normalization,
                    network_batch_normalization=network_batch_normalization)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)

    results = []
    experiment_start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        if (time.time() - experiment_start_time > MAX_EXPERIMENT_TIME_IN_MINUTES * 60): # the experiment took more than MAX_EXPERIMENT_TIME_IN_MINUTES
            break
        epoch_start_time = time.time()
        model.train()

        train_loss_sum = 0
        train_good_sum = 0
        for X, y in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(X)

            # Calculate loss
            loss = loss_function(y_pred, y)

            # Add L1 regularization
            if l1 > 0:
                l1_loss = 0
                for param in model.parameters():
                    l1_loss += torch.norm(param, p=1)
                loss += l1 * l1_loss

            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

            # sum correct predictions
            prediction = torch.round(y_pred)
            correct_pred = (prediction == y).sum().item()
            train_good_sum += correct_pred

        total = len(X_train)

        train_loss = train_loss_sum / total
        train_acc = train_good_sum / total

        model.eval()
        with torch.no_grad():
            val_loss_sum = 0
            val_good_sum = 0
            for X, y in val_dataloader:
                y_pred = model(X)

                # Calculate loss
                loss = loss_function(y_pred, y)

                # Add L1 regularization
                if l1 > 0:
                    l1_loss = 0
                    for param in model.parameters():
                        l1_loss += torch.norm(param, p=1)
                    loss += l1 * l1_loss

                val_loss_sum += loss.item()

                # sum correct predictions
                prediction = torch.round(y_pred)
                correct_pred = (prediction == y).sum().item()
                val_good_sum += correct_pred

            total = len(X_val)

            val_loss = val_loss_sum / total
            val_acc = val_good_sum / total

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