import numpy as np
import _pickle as pickle

seqFile = './resource/mimic4.seqs'
seqs = np.array(pickle.load(open(seqFile, 'rb')))
lengths = np.array([len(seq) for seq in seqs]) - 1
n_samples = len(seqs)
maxlen = np.max(lengths)
print(lengths)
print(n_samples)
print(maxlen)