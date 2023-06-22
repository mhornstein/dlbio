import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split

# Encoding and sequence length constants
ENCODING = {'A': torch.tensor([1, 0, 0, 0]),
            'G': torch.tensor([0, 1, 0, 0]),
            'C': torch.tensor([0, 0, 1, 0]),
            'U': torch.tensor([0, 0, 0, 1]),
            'T': torch.tensor([0, 0, 0, 1]),
            'N': torch.tensor([0.25, 0.25, 0.25, 0.25])}

MAX_LENGTH = 40

class ConvNet(nn.Module):
    def __init__(self, pooling_size):
        super(ConvNet, self).__init__()
        self.pooling_size = pooling_size
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 3 * MAX_LENGTH // pooling_size, 64)
        self.fc2 = nn.Linear(64, 1)

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
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze() # squeeze is used to remove the extra dimension from the output

def encode_and_pad(seq_list):
    encoded_seqs = []
    for seq in seq_list:
        encoded_seq = [ENCODING[ch] for ch in seq]
        while len(encoded_seq) < MAX_LENGTH:
            encoded_seq.append(ENCODING['N'])
        encoded_seqs.append(torch.stack(encoded_seq)[:MAX_LENGTH].t())
    return torch.stack(encoded_seqs)  # We stack the list to create a single tensor


def calculate_accuracy(y_true, y_pred):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_true).sum().float()
    return correct_results_sum/y_true.shape[0]

def split_and_train(X, y, model, num_epochs=10):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        model.train()
        X_train_t = encode_and_pad(X_train)
        y_train_t = torch.tensor(y_train).float()
        optimizer.zero_grad()
        y_pred = model(X_train_t)
        loss = loss_function(y_pred, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            X_val_t = encode_and_pad(X_val)
            y_val_t = torch.tensor(y_val).float()
            y_val_pred = model(X_val_t)
            val_loss = loss_function(y_val_pred, y_val_t)

        train_acc = calculate_accuracy(y_train_t, y_pred)
        val_acc = calculate_accuracy(y_val_t, y_val_pred)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item()}, Training Accuracy: {train_acc}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_acc}')

# Mock data
X = ['AGCTUUAGCTN', 'GTACGTAGCTN', 'TGTACGTAGCT', 'CGTACGTAGCT']
y = [0, 1, 0, 1]

# Initialize the network
model = ConvNet(pooling_size=5)

# Train the network
split_and_train(X, y, model, num_epochs=10)
