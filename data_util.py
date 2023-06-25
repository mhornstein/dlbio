import re
import os
import numpy as np
MAX_SAMPLE_LENGTH = 40
PADDING_CHAR = 'N'
import torch

# Encoding and sequence length constants
ENCODING = {'A': [1, 0, 0, 0],
            'G': [0, 1, 0, 0],
            'C': [0, 0, 1, 0],
            'U': [0, 0, 0, 1],
            'T': [0, 0, 0, 1],
            'N': [0.25, 0.25, 0.25, 0.25]}

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
    elif mode == 'WEIGHTED_LOW':
        for filename in filesnames:
            consentration = get_file_number(filename)
            if consentration == 0: # skip the input file
                continue
            percentage = (total_consentrations - consentration) / total_consentrations
            num_of_samples = set_size * percentage
            lines = read_samples(filename, num_of_samples)
            dataset += lines
            print(f"{consentration}, {percentage}, {num_of_samples}")
    elif mode == 'HIGH':
        filename = filesnames[-1]
        dataset = read_samples(filename, set_size)
    elif mode == 'LOW':
        if 'input' in filesnames[0]: # Skip the first file in case it is input file (with random samples) if possible
            filename = filesnames[0] if len(filesnames) == 1 else filesnames[1]
        else:
            filename = filesnames[0]
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

def create_dataset(rbns_files, mode, set_size):
    '''
    Creates a dataset from the provided RBNS files.
    Parameters:
    - rbns_files (list): A list of RBNS files to use for dataset creation.
    - mode (str): The mode of dataset creation. This can be 'WEIGHTED_LOW', 'WEIGHTED_HIGH', 'LOW', 'HIGH'.
    - set_size (int): The desired size of the positive set (and negative set).

    Returns:
    - samples: A tensor of encoded and padded samples for the dataset. Its shape is 2*set_size X 4 X MAX_SAMPLE_LENGTH
    - labels (torch.Tensor): A tensor of corresponding labels for the samples. Its shape is 2*set_size X 1
    '''
    positive_samples = create_positive_dataset(rbns_files, mode, set_size)
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

    return samples, labels
