import torch
import torch.nn as nn

ONE_HOT = 'one-hot'

VOCAB = ['N', 'A', 'G', 'C', 'T'] # we make sure that every U was replaced with T when loading the RNA compete files
C2I = {nucleotide: i for i, nucleotide in enumerate(VOCAB, start=0)} # this way 'N' will be encoded as 0

# Encoding and sequence length constants
ONE_HOT_ENCODING = { 'N': [0.25, 0.25, 0.25, 0.25],
                     'A': [1, 0, 0, 0],
                     'G': [0, 1, 0, 0],
                     'C': [0, 0, 1, 0],
                     'T': [0, 0, 0, 1] }

def get_embedding_layer(embedding_dim):
    if embedding_dim == ONE_HOT: # return untrainable (frozen) embedding layer representing the one-hot encoding
        embedding_matrix = torch.FloatTensor([ONE_HOT_ENCODING[nucleotide] for nucleotide in VOCAB])
        embeddings = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
    else: # return a trainable embedding layaer
        vocab_size = len(VOCAB)
        embeddings = nn.Embedding(vocab_size, embedding_dim)
    return embeddings
