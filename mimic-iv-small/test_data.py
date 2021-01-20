import numpy as np
import _pickle as pickle

# seqFile = './resource/mimic4.seqs'
seqFile = './mimic.seqs'
# labelFile = './resource/mimic3.labels'
# typeFile = './resource/mimic3.types'
seqs = pickle.load(open(seqFile, 'rb'))
# types = pickle.load(open(typeFile, 'rb'))

seqs = np.array(pickle.load(open(seqFile, 'rb')))
lengths = np.array([len(seq) for seq in seqs]) - 1
n_samples = len(seqs)
maxlen = np.max(lengths)
minlen = np.min(lengths)
print(lengths)
print(n_samples)
print(maxlen)

# labels = np.array(pickle.load(open(labelFile, 'rb')))
# labels = np.expand_dims(labels, axis=1)