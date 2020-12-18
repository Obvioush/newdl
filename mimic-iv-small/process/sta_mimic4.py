import pickle
import numpy as np

if __name__ == '__main__':
    seqs = pickle.load(open('../resource/mimic4.seqs', 'rb'))
    # types = pickle.load(open('../resource/mimic4.types', 'rb'))
    # retype = dict([(v, k) for k, v in types.items()])
    labels = pickle.load(open('../resource/mimic4.allLabels', 'rb'))
    trees = pickle.load(open('../resource/mimic4_newTree.seqs', 'rb'))

    lengths = np.array([len(seq) for seq in seqs]) - 1
    timestep = np.max(lengths)
    # n_samples = len(seqs)
    # visits = np.sum(lengths)
    # avg_visit = visits/n_samples
    #
    # lengths_icd9 = []
    # for patient in seqs:
    #     for visit in patient:
    #         lengths_icd9.append(len(visit))
    #
    #
    # avg_visit_icd9 = np.sum(lengths_icd9) / visits
    # max_visit_icd9 = np.max(lengths_icd9)

    # # label seq
    # label_lengths = np.array([len(seq) for seq in labels])
    # label_samples = len(labels)
    # label_visits = np.sum(label_lengths)
    #
    # lengths_label = []
    # for patient in labels:
    #     for visit in patient:
    #         lengths_label.append(len(visit))
    #
    # avg_label_ccs = np.sum(lengths_label) / label_visits
    # max_label_ccs = np.max(lengths_label)
    #
    # # tree seq
    # tree_lengths = np.array([len(seq) for seq in trees])
    # tree_samples = len(trees)
    # tree_visits = np.sum(tree_lengths)
    #
    # lengths_tree = []
    # for patient in trees:
    #     for visit in patient:
    #         lengths_tree.append(len(visit))
    #
    # avg_tree_ccs = np.sum(lengths_tree) / tree_visits
    # max_tree_ccs = np.max(lengths_tree)