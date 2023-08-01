'''
This utility script provides the functionality of encoding the RNA sequences as ordinal numbers or continous vectors
'''

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
    '''
    Creates an embedding layer used for representing nucleotide sequences.
    It supports two types of embedding layers: "one-hot encoding" for untrainable (frozen) representations
    and "trainable embeddings" for continuous numerical vectors with a specified dimension.

    Parameters:
        embedding_dim: The dimensionality of the embedding vectors.
        Use ONE_HOT for one-hot encoding or any positive integer for trainable embeddings.

    Returns:
        embeddings (torch.nn.Embedding): The embedding layer object for nucleotide sequence representations.
    '''
    if embedding_dim == ONE_HOT: # return untrainable (frozen) embedding layer representing the one-hot encoding
        embedding_matrix = torch.FloatTensor([ONE_HOT_ENCODING[nucleotide] for nucleotide in VOCAB])
        embeddings = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
    else: # return a trainable embedding layaer
        vocab_size = len(VOCAB)
        embeddings = nn.Embedding(vocab_size, embedding_dim)
    return embeddings
