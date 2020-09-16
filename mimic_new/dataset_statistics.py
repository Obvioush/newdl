import numpy as np
import _pickle as pickle

seqFile = './resource/process_data/process.dataseqs'
sequences = np.array(pickle.load(open(seqFile, 'rb')))


lengths = np.array([len(seq) for seq in sequences])
n_samples = len(sequences)
maxlen = np.max(lengths)
minlen = np.min(lengths)
avglen = np.average(lengths)
sumlen = np.sum(lengths)

Visit = []
for seq in sequences:
    a = []
    for visit in seq:
        a.append(len(visit))
    Visit.append(np.array(a))

max = 0
for i in Visit:
    if max < np.max(i):
        max = np.max(i)


Visit2 = []
count = 0
for seq in sequences:
    for visit in seq:
        count = len(visit) + count
    Visit2.append(count)
    count = 0

Visit2 = np.array(Visit2)
sumlenV = np.sum(Visit2)
avgVisits = sumlenV/sumlen