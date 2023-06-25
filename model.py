import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvNet(nn.Module):
    def __init__(self, input_length, hidden_layers, pooling_size, dropout_rate, kernel_sizes, kernels_out_channel,
                 kernel_batch_normalization, network_batch_normalization):
        super(ConvNet, self).__init__()
        self.dropout_rate = dropout_rate
        self.kernel_sizes = kernel_sizes
        self.num_conv_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList()

        in_channels = 4
        out_channels = kernels_out_channel
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
            if kernel_batch_normalization:
                conv_layer = nn.Sequential(conv_layer, nn.BatchNorm1d(out_channels))
            self.conv_layers.append(conv_layer)

        self.dropout = nn.Dropout(self.dropout_rate)

        input_dim = out_channels * self.num_conv_layers * input_length
        if pooling_size == 'Global':
            self.pooling = nn.AdaptiveMaxPool1d(1)
            input_dim //= input_length
        else:
            self.pooling = nn.MaxPool1d(pooling_size)
            input_dim //= pooling_size

        self.hidden_layers = nn.ModuleList()
        for hl_dim in hidden_layers:
            self.hidden_layers.append(nn.Linear(input_dim, hl_dim))
            if network_batch_normalization:
                self.hidden_layers.append(nn.BatchNorm1d(hl_dim))
            input_dim = hl_dim

        self.fc_out = nn.Linear(input_dim, 1) # the output layer is usually kept separate from batch normalization, as it operates differently and does not benefit from the normalization process

    def forward(self, x):
        conv_outputs = []
        for i in range(self.num_conv_layers):
            conv_output = F.relu(self.conv_layers[i](x))
            conv_output = self.pooling(conv_output)
            conv_outputs.append(conv_output)

        x = torch.cat(conv_outputs, dim=1)  # Concatenate along the channel dimension
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
            x = self.dropout(x)

        x = torch.sigmoid(self.fc_out(x))
        return x