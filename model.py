import torch.nn as nn
import torch.nn.functional as F
import torch
import encoding_util

class ConvNet(nn.Module):
    def __init__(self, embedding_dim, input_length, hidden_layers, pooling_size, dropout_rate, kernel_size, stride, kernels_out_channel,
                 kernel_batch_normalization, network_batch_normalization):
        super(ConvNet, self).__init__()
        self.embeddings = encoding_util.get_embedding_layer(embedding_dim)

        self.dropout_rate = dropout_rate

        in_channels = self.embeddings.embedding_dim
        out_channels = kernels_out_channel
        padding = kernel_size // 2
        conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        if kernel_batch_normalization:
            self.conv_layer = nn.Sequential(conv_layer, nn.BatchNorm1d(out_channels))
        else:
            self.conv_layer = conv_layer

        self.dropout = nn.Dropout(self.dropout_rate)

        if pooling_size == 'Global':
            # for global pooling, set L_out to be 1. Documentation: https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool1d.html
            L_out = 1
            self.pooling = nn.AdaptiveMaxPool1d(L_out)
        else:
            self.pooling = nn.MaxPool1d(pooling_size)
            # General formula for CNN output size appear in the documentation: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            # after substituting our parameters we get: L_out = (input_length - 1) // stride + 1
            L_out = (input_length - 1) // stride + 1
            L_out //= pooling_size

        input_dim = out_channels * L_out

        self.hidden_layers = nn.ModuleList()
        for hl_dim in hidden_layers:
            self.hidden_layers.append(nn.Linear(input_dim, hl_dim))
            if network_batch_normalization:
                self.hidden_layers.append(nn.BatchNorm1d(hl_dim))
            input_dim = hl_dim

        self.fc_out = nn.Linear(input_dim, 1) # the output layer is usually kept separate from batch normalization, as it operates differently and does not benefit from the normalization process

    def forward(self, x):
        embs = self.embeddings(x)

        # transpose: SET_SIZE X MAX_SAMPLE_LENGTH X ENCODING_LENGTH => SET_SIZE X ENCODING_LENGTH X MAX_SAMPLE_LENGTH
        # e.g. for one-hot encoded embeddings: batch_size X 40 X 4 => batch_size X 4 X 40
        x = torch.transpose(embs, -1, -2)

        x = F.relu(self.conv_layer(x))
        x = self.pooling(x)

        x = x.view(x.size(0), -1)

        x = self.dropout(x)

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
            x = self.dropout(x)

        x = torch.sigmoid(self.fc_out(x))
        return x