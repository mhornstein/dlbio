import sys
import re
import os

MODE = 'WEIGHTED_HIGH' # 'WEIGHTED_LOW', 'WEIGHTED_HIGH', 'LOW', 'HIGH'
SET_SIZE = 100

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

def create_porisive_dataset(mode, files, set_size):
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

if __name__ == '__main__':
    rna_compete_filename = sys.argv[1]
    rna_compete_sequences = load_rna_compete(rna_compete_filename)
    
    rbns_files = sys.argv[2:]
    positive_samples = create_porisive_dataset(MODE, rbns_files, SET_SIZE)
    # negative_samples = shuffle_samples(positive_samples)
