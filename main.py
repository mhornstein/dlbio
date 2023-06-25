import sys
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import time
from model import ConvNet
from data_util import load_rna_compete, create_dataset
import torch

MODE = 'WEIGHTED_HIGH' # 'WEIGHTED_LOW', 'WEIGHTED_HIGH', 'LOW', 'HIGH'
SET_SIZE = 100
BATCH_SIZE = 64

def create_data_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def calculate_accuracy(y_true, y_pred):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_true).sum().float()
    return correct_results_sum/y_true.shape[0]

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
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        train_acc = 0
        for X, y in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += calculate_accuracy(y, y_pred)

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            for X, y in val_dataloader:
                y_pred = model(X)
                loss = loss_function(y_pred, y)
                val_loss += loss.item()
                val_acc += calculate_accuracy(y, y_pred)

            val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader)

        epoch_time = time.time() - start_time
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.5f}, Training Accuracy: {train_acc:.5f}, Validation Loss: {val_loss:.5f}, Validation Accuracy: {val_acc:.5f}, time: {epoch_time:.2f} seconds')