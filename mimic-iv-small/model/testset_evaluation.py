import tensorflow as tf
import tensorflow.keras as keras
import _pickle as pickle
import numpy as np
import heapq
import operator
import os

_TEST_RATIO = 0.15
_VALIDATION_RATIO = 0.1
codeCount = 6534  # icd9数
labelCount = 277  # 标签的类别数
treeCount = 728  # 分类树的祖先节点数量
timeStep = 145


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


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


def codel_level_accuracy(y_true, y_pred, rank=[5, 10, 15, 20, 25, 30]):
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


def kame_knowledgematrix(treeseq_set):
    # 和患者输入保持一致，访问为1到n-1
    for i in range(len(treeseq_set)):
        treeseq_set[i] = treeseq_set[i][:-1]

    zerovec = np.zeros((87, 728), dtype=np.int8)
    ts = []
    for i in treeseq_set:
        count = 0
        a = []
        for j in i:
            # 变为onehot
            temp = keras.utils.to_categorical(j, dtype=np.int8)
            if len(temp) < 87:
                zerovec1 = np.zeros((87-len(temp), 728), dtype=np.int8)
                temp = np.r_[temp, zerovec1]
            count += 1
            a.append(temp)
        while count < timeStep:
            a.append(zerovec)
            count += 1
        ts.append(a)

    ts = np.array(ts)
    return ts



if __name__ == '__main__':
    seqFile = '../resource/mimic4.seqs'
    labelFile = '../resource/mimic4.allLabels'
    treeFile = '../resource/mimic4_newTree.seqs'

    gloveKnowledgeFile = '../resource/gram_emb/mimic4_glove_knowledge_emb.npy'
    # node2vecFile = './resource/embedding/node2vec_test.npy'
    gramembFile = '../resource/gram_emb/gramemb_diagcode.npy'

    train_set, valid_set, test_set = load_data(seqFile, labelFile, treeFile)
    x_test, y_test, tree_test = padMatrix(test_set[0], test_set[1], test_set[2])

    # RNN模型
    # model = tf.keras.models.load_model('G:\\mimic4_small_model_save\\model_RNN\\RNN_128\\RNN_epoch_40')

    # RNN+模型
    # model = tf.keras.models.load_model('G:\\mimic4_small_model_save\\model_RNN+\\RNN+_128\\RNN+_epoch_38')

    # Dipole模型
    # model = tf.keras.models.load_model('G:\mimic4_small_model_save\model_Dipole\Dipole_new_128\Dipole_epoch_70')

    # GRAM模型
    # model = tf.keras.models.load_model('G:\\mimic4_small_model_save\\model_GRAM\\GRAM_128\\GRAM_epoch_53')

    # KAME模型
    # tree_test = kame_knowledgematrix(test_set[2])
    # model = tf.keras.models.load_model('G:\\mimic4_small_model_save\\model_KAME\\KAME_128\\KAME_epoch_25')

    # 我们的模型
    # model = tf.keras.models.load_model('G:\\mimic4_small_model_save\\model_NKAM\\NKAM_128\\NKAM_epoch_90')
    # model = tf.keras.models.load_model('G:\\mimic4_small_model_save\\model_NEW\\NEW_128\\NEW_epoch_90')


    # filePath = 'G:\\mimic4_small_model_save\\model_GCSAM\\test_head\head_1\\NEW_epoch_100'
    # filePath = 'G:\\mimic4_small_model_save\\model_GCSAM\\test_head\head_3\\NEW_epoch_100'
    # filePath = 'G:\\mimic4_small_model_save\\model_GCSAM\\test_head\head_4\\NEW_epoch_100'
    # filePath = 'G:\\mimic4_small_model_save\\model_GCSAM\\test_head\head_5\\NEW_epoch_100'
    # filePath = 'G:\\mimic4_small_model_save\\model_GCSAM\\test_head\head_6\\NEW_epoch_99'
    # filePath = 'G:\\mimic4_small_model_save\\model_GCSAM\\test_head\head_7\\NEW_epoch_99'
    filePath = 'G:\\mimic4_small_model_save\\model_GCSAM\\test_head\head_8\\NEW_epoch_99'
    model = tf.keras.models.load_model(filePath)

    # RNN、RNN+、Dipole、GRAM模型的预测
    # preds = model.predict(x_test, batch_size=100)

    # KAME、NKAM模型的预测
    preds = model.predict([x_test, tree_test], batch_size=100)

    y_pred = convert2preds(preds)
    y_true = process_label(test_set[1])
    metrics_visit_level_precision = visit_level_precision(y_true, y_pred)
    metrics_codel_level_accuracy = codel_level_accuracy(y_true, y_pred)

    print("Top-5 precision为：", metrics_visit_level_precision[0])
    print("Top-10 precision为：", metrics_visit_level_precision[1])
    print("Top-15 precision为：", metrics_visit_level_precision[2])
    print("Top-20 precision为：", metrics_visit_level_precision[3])
    print("Top-25 precision为：", metrics_visit_level_precision[4])
    print("Top-30 precision为：", metrics_visit_level_precision[5])
    print("---------------------------------------------------------")
    print("Top-5 Recall为：", metrics_codel_level_accuracy[0])
    print("Top-10 Recall为：", metrics_codel_level_accuracy[1])
    print("Top-15 Recall为：", metrics_codel_level_accuracy[2])
    print("Top-20 Recall为：", metrics_codel_level_accuracy[3])
    print("Top-25 Recall为：", metrics_codel_level_accuracy[4])
    print("Top-30 Recall为：", metrics_codel_level_accuracy[5])
