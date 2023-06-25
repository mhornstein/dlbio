import sys
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import time
from model import ConvNet
from data_util import load_rna_compete, create_dataset
import torch
import pandas as pd

MODE = 'WEIGHTED_HIGH' # 'WEIGHTED_LOW', 'WEIGHTED_HIGH', 'LOW', 'HIGH'
SET_SIZE = 100
BATCH_SIZE = 64
EPS = 1e-12

def create_data_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def calculate_accuracy(y_true, y_pred):
    correct_results_sum = (y_pred == y_true).sum().float()
    return (correct_results_sum/y_true.shape[0]).item()

def calculate_f1_score(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum().float()
    fp = ((y_pred == 1) & (y_true == 0)).sum().float()
    fn = ((y_pred == 0) & (y_true == 1)).sum().float()
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * (precision * recall) / (precision + recall + EPS)
    return f1.item()

if __name__ == '__main__':
    learning_rate = 0.01
    num_epochs = 10
    rna_compete_filename = sys.argv[1]
    rna_compete_sequences = load_rna_compete(rna_compete_filename)
    
    rbns_files = sys.argv[2:]
    samples, labels = create_dataset(rbns_files, MODE, SET_SIZE)
    input_length = samples.shape[-1]

    X_train, X_val, y_train, y_val = train_test_split(samples, labels, test_size=0.2, random_state=42)
    train_dataloader = create_data_loader(X_train, y_train, BATCH_SIZE, True)
    val_dataloader = create_data_loader(X_val, y_val, BATCH_SIZE, False)

    model = ConvNet(input_length=input_length, hidden_layers=[32, 64], pooling_size=5, dropout_rate=0.2, kernel_sizes=[7, 15], kernels_out_channel=64,
                    kernel_batch_normalization=True, network_batch_normalization=True)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    results = []
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0
        train_acc = 0
        train_f1 = 0
        for X, y in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(X)

            # Calculate loss
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate classification metrics. change predicions from probability -> label
            y_pred = torch.round(y_pred)
            train_acc += calculate_accuracy(y, y_pred)
            train_f1 += calculate_f1_score(y, y_pred)

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        train_f1 /= len(train_dataloader)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            val_f1 = 0
            for X, y in val_dataloader:
                y_pred = model(X)

                # Calculate loss
                loss = loss_function(y_pred, y)
                val_loss += loss.item()

                # Calculate classification metrics. change predicions from probability -> label
                y_pred = torch.round(y_pred)
                val_acc += calculate_accuracy(y, y_pred)
                val_f1 += calculate_f1_score(y, y_pred)

            val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader)
            val_f1 /= len(val_dataloader)

        epoch_time = time.time() - start_time

        result_entry = {'epoch': epoch,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'train_f1': train_f1,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_f1': val_f1,
                        'epoch_time': epoch_time }
        results.append(result_entry)

        print(f'Epoch {epoch}/{num_epochs}, '
              f'Train Loss: {train_loss:.5f}, Train acc: {train_acc:.5f}, Train F1: {train_f1:.5f}, '
              f'Val Loss: {val_loss:.5f}, Val acc: {val_acc:.5f}, Val F1: {val_f1:.5f}, '
              f'time: {epoch_time:.2f} seconds')

    print(pd.DataFrame(results).set_index('epoch'))