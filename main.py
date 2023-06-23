import sys
import re
import os
from torch.utils.data import TensorDataset, DataLoader
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, ZeroPadding2D
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split

MODE = 'WEIGHTED_HIGH' # 'WEIGHTED_LOW', 'WEIGHTED_HIGH', 'LOW', 'HIGH'
SET_SIZE = 100
MAX_SAMPLE_LENGTH = 40
PADDING_CHAR = 'N'
FILTERS = 32
KERNEL_SIZE = (8, 4)
POOL_SIZE = (2,2)
HIDDEN_LAYERS_DIMS = [128, 64, 1]
BATCH_SIZE = 64

# Encoding and sequence length constants
ENCODING = {'A': [1, 0, 0, 0],
            'G': [0, 1, 0, 0],
            'C': [0, 0, 1, 0],
            'U': [0, 0, 0, 1],
            'T': [0, 0, 0, 1],
            'N': [0.25, 0.25, 0.25, 0.25]}

class ConvNet(nn.Module):
    def __init__(self,hidden_layers, pooling_size, dropout_rate):
        super(ConvNet, self).__init__()
        self.pooling_size = pooling_size
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.hidden_layers = nn.ModuleList()
        input_dim = 32 * 3 * MAX_SAMPLE_LENGTH // pooling_size
        for hl_dim in hidden_layers:
            self.hidden_layers.append(nn.Linear(input_dim, hl_dim))
            input_dim = hl_dim

        self.fc_out = nn.Linear(input_dim, 1)


    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x1 = F.max_pool1d(x1, self.pooling_size)

        x2 = F.relu(self.conv2(x))
        x2 = F.avg_pool1d(x2, self.pooling_size)

        x3 = F.relu(self.conv3(x))
        x3 = F.max_pool1d(x3, self.pooling_size)

        x = torch.cat((x1, x2, x3), dim=-1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
            x = self.dropout(x)

        x = torch.sigmoid(self.fc_out(x))
        return x


def load_rna_compete(rna_compete_filename):
    seqs = []
    with open(rna_compete_filename) as f:
        for line in f:
            seqs.append(line.strip())
    return seqs

def get_file_number(file_path):
    file_name = os.path.basename(file_path)
    if 'input' in file_name:
        return 0
    else:
        numeric_value = re.search(r'_(\d+)n', file_name).group(1)
        return int(numeric_value)

def read_samples(file_path, num_of_samples):
    lines = []
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            if count >= num_of_samples:
                break
            seq = line.strip().split()[0]
            lines.append(seq)
            count += 1
    return lines

def create_positive_dataset(files, mode, set_size):
    dataset = []
    filesnames = sorted(files, key=get_file_number)
    total_consentrations = sum([get_file_number(file_path) for file_path in files])
    if mode == 'WEIGHTED_HIGH':
        for filename in filesnames:
            consentration = get_file_number(filename)
            percentage = consentration / total_consentrations
            num_of_samples = set_size * percentage
            lines = read_samples(filename, num_of_samples)
            dataset += lines
    elif mode == 'HIGH':
        filename = filesnames[-1]
        dataset = read_samples(filename, set_size)
    else:
        raise ValueError(f'Unknown mode: {mode}')

    return dataset

def shuffle_samples(samples):
    shuffled_seqs = [''.join(np.random.permutation(list(seq))) for seq in samples]
    return shuffled_seqs

def encode_sequence_list(seq_list, encoding):
    encoded_seqs = []
    for seq in seq_list:
        encoded_seq = [encoding[ch] for ch in seq]
        encoded_seqs.append(encoded_seq)
    tensor = torch.tensor(encoded_seqs)
    tensor = torch.transpose(tensor, -1, -2) # transpose: SET_SIZE X MAX_SAMPLE_LENGTH X 4 => SET_SIZE X 4 X MAX_SAMPLE_LENGTH
    return tensor

def pad_samples(samples, max_length, padding_char):
    padded_samples = list(map(lambda s: s[:max_length].ljust(max_length, padding_char), samples))
    return padded_samples

def create_model(input_shape, filters, kernel_size, pool_size, hidden_layers_dims):
    padding_shape = (kernel_size[0] - 1, kernel_size[1])

    input_layer = Input(shape=input_shape)
    input_layer_padded = ZeroPadding2D(padding=padding_shape)(input_layer)
    conv_layer = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(input_layer_padded)
    max_pool_layer = MaxPooling2D(pool_size=pool_size)(conv_layer)
    flatten_layer = Flatten()(max_pool_layer)

    last_layer = flatten_layer
    for dim in hidden_layers_dims[:-1]:
        last_layer = Dense(dim, activation='relu')(last_layer)
    output_layer = Dense(1, activation='sigmoid')(last_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy()])

    return model


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
    positive_samples = create_positive_dataset(rbns_files, MODE, SET_SIZE)
    negative_samples = shuffle_samples(positive_samples)

    positive_samples = pad_samples(positive_samples, MAX_SAMPLE_LENGTH, PADDING_CHAR)
    negative_samples = pad_samples(negative_samples, MAX_SAMPLE_LENGTH, PADDING_CHAR)

    positive_labels = [1] * len(positive_samples)
    negative_labels = [0] * len(negative_samples)

    samples = encode_sequence_list(positive_samples + negative_samples, ENCODING)
    labels = torch.FloatTensor(positive_labels + negative_labels).reshape(-1, 1)

    # shuffling the data
    index = np.arange(samples.shape[0]) # Create an index array and shuffle it
    np.random.shuffle(index)
    samples = samples[index]
    labels = labels[index]

    X_train, X_val, y_train, y_val = train_test_split(samples, labels, test_size=0.2, random_state=42)
    train_dataloader = create_data_loader(X_train, y_train, BATCH_SIZE, True)
    val_dataloader = create_data_loader(X_val, y_val, BATCH_SIZE, False)

    model = ConvNet(hidden_layers=[32, 64], pooling_size=5, dropout_rate=0.2)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    for epoch in range(num_epochs):
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

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}, Training Accuracy: {train_acc}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')