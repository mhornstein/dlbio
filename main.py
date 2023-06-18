import sys
import re
import os
import numpy as np

MODE = 'WEIGHTED_HIGH' # 'WEIGHTED_LOW', 'WEIGHTED_HIGH', 'LOW', 'HIGH'
SET_SIZE = 100
MAX_SAMPLE_LENGTH = 41
PADDING_CHAR = 'N'

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

def read_samples(file_path, num_of_samples, max_sample_length, padding_char):
    lines = []
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            if count >= num_of_samples:
                break
            seq = line.strip().split()[0]
            seq = seq[:max_sample_length] if len(seq) > max_sample_length else seq.ljust(max_sample_length, padding_char)
            lines.append(seq)
            count += 1
    return lines

def create_positive_dataset(mode, files, set_size, max_sample_length, padding_char):
    dataset = []
    filesnames = sorted(files, key=get_file_number)
    total_consentrations = sum([get_file_number(file_path) for file_path in files])
    if mode == 'WEIGHTED_HIGH':
        for filename in filesnames:
            consentration = get_file_number(filename)
            percentage = consentration / total_consentrations
            num_of_samples = set_size * percentage
            lines = read_samples(filename, num_of_samples, max_sample_length, padding_char)
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

if __name__ == '__main__':
    rna_compete_filename = sys.argv[1]
    rna_compete_sequences = load_rna_compete(rna_compete_filename)
    
    rbns_files = sys.argv[2:]
    positive_samples = create_positive_dataset(MODE, rbns_files, SET_SIZE, MAX_SAMPLE_LENGTH, PADDING_CHAR)
    negative_samples = shuffle_samples(positive_samples)

    positive_labels = [1] * len(positive_samples)
    negative_labels = [0] * len(positive_samples)

    samples = encode_sequence_list(positive_samples + negative_samples, ENCODING)
    labels = np.array(positive_labels + negative_labels).reshape(-1, 1)

    # shuffling the data
    index = np.arange(samples.shape[0]) # Create an index array and shuffle it
    np.random.shuffle(index)

    samples = samples[index]
    labels = labels[index]

