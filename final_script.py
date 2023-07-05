'''
Input: 
    The 1st argument is the RNAcompete filename, and 4-6 filenames of RBNS files
Output:
    a file with RNA binding intensities (in the same order of the RNA sequences)
Flow:
    1. Parse RBNS files.
    2. create positive + negative examples.
    3. train the model.
    4. Get model classification on rna_compete_file intensities
    5. Create resulsts file. 
'''

