import sys
import re
import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, ZeroPadding2D
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy

MODE = 'WEIGHTED_HIGH' # 'WEIGHTED_LOW', 'WEIGHTED_HIGH', 'LOW', 'HIGH'
SET_SIZE = 100
MAX_SAMPLE_LENGTH = 40
PADDING_CHAR = 'N'
FILTERS = 32
KERNEL_SIZE = (8, 4)
POOL_SIZE = (2,2)
HIDDEN_LAYERS_DIMS = [128, 64, 1]

ENCODING = {'A': np.array([1, 0, 0, 0]), 'G': np.array([0, 1, 0, 0]),
            'C': np.array([0, 0, 1, 0]), 'U': np.array([0, 0, 0, 1]),
            'T': np.array([0, 0, 0, 1]),
            'N': np.array([0.25] * 4)}

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
    else:
        raise ValueError(f'Unknown mode: {mode}')

    return dataset

def shuffle_samples(samples):
    shuffled_seqs = [''.join(np.random.permutation(list(seq))) for seq in samples]
    return shuffled_seqs

def encode_sequence_list(sequence_list, encoding):
    encoded_sequences = [np.array([encoding[base] for base in sequence]) for sequence in sequence_list]
    return np.array(encoded_sequences)

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

if __name__ == '__main__':
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
    labels = np.array(positive_labels + negative_labels).reshape(-1, 1)

    # shuffling the data
    index = np.arange(samples.shape[0]) # Create an index array and shuffle it
    np.random.shuffle(index)
    samples = samples[index]
    labels = labels[index]

    model = create_model(input_shape=samples.shape[1:], filters=FILTERS,
                         kernel_size=KERNEL_SIZE, pool_size=POOL_SIZE, hidden_layers_dims=HIDDEN_LAYERS_DIMS)
    print(f'Model details:\n{model.summary()}')

