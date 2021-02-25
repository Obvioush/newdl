import tensorflow as tf
import pickle
import numpy as np
import heapq
import operator

_TEST_RATIO = 0.15
_VALIDATION_RATIO = 0.1
codeCount = 4880  # icd9数
labelCount = 272  # 标签的类别数
treeCount = 728  # 分类树的祖先节点数量
timeStep = 41

def load_data(seqFile, labelFile, treeFile=''):
    sequences = np.array(pickle.load(open(seqFile, 'rb')))
    labels = np.array(pickle.load(open(labelFile, 'rb')))
    if len(treeFile) > 0:
        trees = np.array(pickle.load(open(treeFile, 'rb')))

    np.random.seed(0)
    dataSize = len(labels)
    ind = np.random.permutation(dataSize)
    nTest = int(_TEST_RATIO * dataSize)
    nValid = int(_VALIDATION_RATIO * dataSize)

    test_indices = ind[:nTest]
    valid_indices = ind[nTest:nTest+nValid]
    train_indices = ind[nTest+nValid:]

    train_set_x = sequences[train_indices]
    train_set_y = labels[train_indices]
    test_set_x = sequences[test_indices]
    test_set_y = labels[test_indices]
    valid_set_x = sequences[valid_indices]
    valid_set_y = labels[valid_indices]
    train_set_t = None
    test_set_t = None
    valid_set_t = None

    if len(treeFile) > 0:
        train_set_t = trees[train_indices]
        test_set_t = trees[test_indices]
        valid_set_t = trees[valid_indices]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]

    if len(treeFile) > 0:
        train_set_t = [train_set_t[i] for i in train_sorted_index]
        valid_set_t = [valid_set_t[i] for i in valid_sorted_index]
        test_set_t = [test_set_t[i] for i in test_sorted_index]

    train_set = (train_set_x, train_set_y, train_set_t)
    valid_set = (valid_set_x, valid_set_y, valid_set_t)
    test_set = (test_set_x, test_set_y, test_set_t)

    return train_set, valid_set, test_set


def padMatrix(seqs, labels, treeseqs=''):
    # lengths = np.array([len(seq) for seq in seqs]) - 1
    n_samples = len(seqs)
    # maxlen = np.max(lengths)

    x = np.zeros((n_samples, timeStep, codeCount), dtype=np.int8)
    y = np.zeros((n_samples, timeStep, labelCount), dtype=np.int8)

    if len(treeseqs) > 0:
        tree = np.zeros((n_samples, timeStep, treeCount), dtype=np.int8)
        for idx, (seq, lseq, tseq) in enumerate(zip(seqs, labels, treeseqs)):
            for xvec, subseq in zip(x[idx, :, :], seq[:-1]):
                xvec[subseq] = 1.
            for yvec, subseq in zip(y[idx, :, :], lseq[1:]):
                yvec[subseq] = 1.
            for tvec, subseq in zip(tree[idx, :, :], tseq[:-1]):
                tvec[subseq] = 1.
        return x, y, tree

    else:
        for idx, (seq, lseq) in enumerate(zip(seqs, labels)):
            for xvec, subseq in zip(x[idx, :, :], seq[:-1]):
                xvec[subseq] = 1.
            for yvec, subseq in zip(y[idx, :, :], lseq[1:]):
                yvec[subseq] = 1.
        return x, y


def calculate_dimSize(seqFile):
    seqs = pickle.load(open(seqFile, 'rb'))
    codeSet = set()
    for patient in seqs:
        for visit in patient:
            for code in visit:
                codeSet.add(code)
    return max(codeSet) + 1


# 为评估函数准备标签序列，从原始序列第二个元素开始
def process_label(labelSeqs):
    newlabelSeq = []
    for i in range(len(labelSeqs)):
        newlabelSeq.append(labelSeqs[i][1:])
    return newlabelSeq


def visit_level_precision(y_true, y_pred, rank=[5, 10, 15, 20, 25, 30]):
    recall = list()
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            thisOne = list()
            codes = y_true[i][j]
            tops = y_pred[i][j]
            for rk in rank:
                thisOne.append(len(set(codes).intersection(set(tops[:rk]))) * 1.0 / min(rk, len(set(codes))))
            recall.append(thisOne)
    return (np.array(recall)).mean(axis=0).tolist()


def code_level_accuracy(y_true, y_pred, rank=[5, 10, 15, 20, 25, 30]):
    recall = list()
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            thisOne = list()
            codes = y_true[i][j]
            tops = y_pred[i][j]
            for rk in rank:
                thisOne.append(len(set(codes).intersection(set(tops[:rk]))) * 1.0 / len(set(codes)))
            recall.append(thisOne)
    return (np.array(recall)).mean(axis=0).tolist()


# 按从大到小取预测值中前30个ccs分组号
def convert2preds(preds):
    ccs_preds = []
    for i in range(len(preds)):
        temp = []
        for j in range(len(preds[i])):
            temp.append(list(zip(*heapq.nlargest(30, enumerate(preds[i][j]), key=operator.itemgetter(1))))[0])
        ccs_preds.append(temp)
    return ccs_preds

if __name__ == '__main__':
    seqFile = '../resource/mimic3.seqs'
    labelFile = '../resource/mimic3.allLabels'
    treeFile = '../resource/mimic3_newTree.seqs'

    train_set, valid_set, test_set = load_data(seqFile, labelFile, treeFile)
    x_test, y_test, tree_test = padMatrix(test_set[0], test_set[1], test_set[2])

    # filePath = 'G:\\mimic3_model_save\\model_GCSAM\\emb_glove\\NEW_epoch_73'
    # filePath = 'G:\\mimic3_model_save\\model_GCSAM\\emb_no\\NEW_epoch_37'
    # filePath = 'G:\\mimic3_model_save\\model_GCSAM\\test_ka\\NEW_epoch_62'
    # filePath = 'G:\\mimic3_model_save\\model_GCSAM\\test_sa\\NEW_epoch_70'
    # filePath = 'G:\\mimic3_model_save\\model_GCSAM\\test_none\\NEW_epoch_26'
    # filePath = 'G:\\mimic3_model_save\\model_GCSAM\\test_head\head_1\\NEW_epoch_97'
    # filePath = 'G:\\mimic3_model_save\\model_GCSAM\\test_head\head_3\\NEW_epoch_80'
    # filePath = 'G:\\mimic3_model_save\\model_GCSAM\\test_head\head_4\\NEW_epoch_96'
    # filePath = 'G:\\mimic3_model_save\\model_GCSAM\\test_head\head_5\\NEW_epoch_96'
    # filePath = 'G:\\mimic3_model_save\\model_GCSAM\\test_head\head_6\\NEW_epoch_84'
    # filePath = 'G:\\mimic3_model_save\\model_GCSAM\\test_head\head_7\\NEW_epoch_75'
    filePath = 'G:\\mimic3_model_save\\model_GCSAM\\test_head\head_8\\NEW_epoch_55'

    # model = tf.keras.models.load_model('G:\\mimic3_model_save\\model_NKAM\\NKAM_new_128\\NKAM_epoch_57')
    # model = tf.keras.models.load_model('G:\\mimic3_model_save\\model_NKAM_alpha\\NKAM_128\\NKAM_alpha_epoch_24')
    # model = tf.keras.models.load_model('G:\\mimic3_model_save\\model_NKAM_belta\\NKAM_128\\NKAM_belta_epoch_54')
    # model = tf.keras.models.load_model('G:\\mimic3_model_save\\model_NKAM_gamma\\NKAM_new_128\\NKAM_gamma_epoch_49')
    # model = tf.keras.models.load_model('G:\\mimic3_model_save\\model_NEW\\NEW_128\\NEW_epoch_98')
    model = tf.keras.models.load_model(filePath)
    preds = model.predict([x_test, tree_test], batch_size=100)

    y_pred = convert2preds(preds)
    y_true = process_label(test_set[1])
    metrics_visit_level_precision = visit_level_precision(y_true, y_pred)
    metrics_code_level_accuracy = code_level_accuracy(y_true, y_pred)

    print("Top-5 visit-level precision为：", metrics_visit_level_precision[0])
    print("Top-10 visit-level precision为：", metrics_visit_level_precision[1])
    print("Top-15 visit-level precision为：", metrics_visit_level_precision[2])
    print("Top-20 visit-level precision为：", metrics_visit_level_precision[3])
    print("Top-25 visit-level precision为：", metrics_visit_level_precision[4])
    print("Top-30 visit-level precision为：", metrics_visit_level_precision[5])
    print("-------------------------------------------------------------------------")
    print("Top-5 code-level accuracy为：", metrics_code_level_accuracy[0])
    print("Top-10 code-level accuracy为：", metrics_code_level_accuracy[1])
    print("Top-15 code-level accuracy为：", metrics_code_level_accuracy[2])
    print("Top-20 code-level accuracy为：", metrics_code_level_accuracy[3])
    print("Top-25 code-level accuracy为：", metrics_code_level_accuracy[4])
    print("Top-30 code-level accuracy为：", metrics_code_level_accuracy[5])