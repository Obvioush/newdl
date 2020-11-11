import pickle
from functools import reduce

if __name__ == '__main__':
    outFile1 = '../resource/mimic3_all'

    treeSeqs = pickle.load(open('../resource/mimic3_tree.seqs', 'rb'))
    seqs = pickle.load(open('../resource/mimic3.seqs', 'rb'))
    # retype = dict([(v, k) for k, v in trees_type.items()])

    # for i in range(len(seqs))
    newSeqs = pickle.load(open('../resource/mimic3.seqs', 'rb'))
    for i in range(len(newSeqs)):
        for j in range(len(newSeqs[i])):
            newSeqs[i][j].extend(treeSeqs[i][j])


    # pickle.dump(newSeqs, open(outFile1 + '.seqs', 'wb'), -1)
