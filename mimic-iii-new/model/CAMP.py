import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import pickle
import numpy as np
import heapq
import operator
import os

from utils import *

_TEST_RATIO = 0.15
_VALIDATION_RATIO = 0.1
gru_dimentions = 128
codeCount = 4880  # icd9数
labelCount = 272  # 标签的类别数
treeCount = 728  # 分类树的祖先节点数量
timeStep = 41


# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'


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


class coAttention(keras.layers.Layer):
    def __init__(self, units, sta_dimention):
        super(coAttention, self).__init__()
        self.units = units
        self.d3 = sta_dimention

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.Wp = self.add_weight(name='Wp',
                                 shape=(self.d3, input_shape[1][1]),
                                 initializer='uniform',
                                 trainable=True)
        self.bp = self.add_weight(name='bp',
                                  shape=(self.d3,),
                                  initializer='uniform',
                                  trainable=True)
        self.W3 = self.add_weight(name='W3',
                                  shape=(self.d3, input_shape[2][2]),
                                  initializer='uniform',
                                  trainable=True)
        self.W4 = self.add_weight(name='W4',
                                  shape=(self.d3, self.d3),
                                  initializer='uniform',
                                  trainable=True)
        self.b3 = self.add_weight(name='b3',
                                  shape=(self.d3,),
                                  initializer='uniform',
                                  trainable=True)
        super(memoryAttention, self).build(input_shape)  # 一定要在最后调用它


    def call(self, inputs):
        K, Q, M, ht = inputs
        q = tf.matmul(self.Wp, Q) + self.bp
        concat = keras.layers.concatenate([ht, q])
        mlp = keras.layers.Dense(self.units)(concat)

        scores = tf.matmul(K, mlp, transpose_b=True)
        distribution = tf.nn.softmax(scores)

        Belta = tf.nn.relu(tf.matmul(self.W3, M)+tf.matmul(self.W4, q)+self.b3)

        q_new = tf.multiply(Belta, q)

        return q_new


class memoryAttention(keras.layers.Layer):
    def __init__(self, units, v_dimention):
        super(memoryAttention, self).__init__()
        self.units = units
        self.dv = v_dimention

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.W1 = self.add_weight(name='W1',
                                 shape=(self.dv, input_shape[1][2]),
                                 initializer='uniform',
                                 trainable=True)
        self.b1 = self.add_weight(name='b1',
                                  shape=(self.dv,),
                                  initializer='uniform',
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(self.dv, input_shape[1][2]),
                                  initializer='uniform',
                                  trainable=True)
        self.b2 = self.add_weight(name='b2',
                                  shape=(self.dv,),
                                  initializer='uniform',
                                  trainable=True)
        self.E = self.add_weight(name='E',
                                  shape=(18, self.dv),
                                  initializer=keras.initializers.ones,
                                  trainable=False)

        super(memoryAttention, self).build(input_shape)  # 一定要在最后调用它


    def call(self, inputs):
        K, ht, V = inputs
        # Lt.shape:(batch_size, length, size, units)=>(5250,41,84,128)
        # rnn_ht.shape:(batch_size, length, units) =>(5250,41,128)

        # ht.shape:(batch_size, length, size, units)=>(5250,41,1,128)
        mlp_ht = keras.layers.Dense(self.units)(ht)
        # score.shape:(batch_size, length, size, 1)
        scores = tf.matmul(K, mlp_ht, transpose_b=True)

        # (Alpha)attention_weights.shape:(batch_size, length, size, 1)
        distribution = tf.nn.softmax(scores)

        erase = tf.nn.sigmoid(tf.matmul(self.W1, ht) + self.b1)
        add = tf.nn.tanh(tf.matmul(self.W2, ht) + self.b2)
        V = tf.multiply(V, (self.E - tf.matmul(distribution, erase))) + tf.matmul(distribution, add)
        # context_vector.shape: (batch_size, length, size, units)
        # tensorflow自己做扩展，把Alpha的1扩展为units
        M = tf.matmul(distribution, V)

        # 在size维度求和，context_vector.shape:(batch_size, length, units)
        M = tf.reduce_sum(M, axis=2)

        return M


def visit_level_precision(y_true, y_pred, rank=[5]):
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


def code_level_accuracy(y_true, y_pred, rank=[5]):
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

# 为评估函数准备标签序列，从原始序列第二个元素开始
def process_label(labelSeqs):
    newlabelSeq = []
    for i in range(len(labelSeqs)):
        newlabelSeq.append(labelSeqs[i][1:])
    return newlabelSeq

# 按从大到小取预测值中前30个ccs分组号
def convert2preds(preds):
    ccs_preds = []
    for i in range(len(preds)):
        temp = []
        for j in range(len(preds[i])):
            temp.append(list(zip(*heapq.nlargest(30, enumerate(preds[i][j]), key=operator.itemgetter(1))))[0])
        ccs_preds.append(temp)
    return ccs_preds


class metricsHistory(Callback):
    def __init__(self):
        super().__init__()
        self.Recall_5 = []
        self.Precision_5 = []
        # self.path = 'G:\\模型训练保存\\ourmodel_' + str(gru_dimentions) + '_dropout\\rate05_02\\'
        # self.fileName = 'model_metrics.txt'

    def on_epoch_end(self, epoch, logs={}):
        # precision5 = visit_level_precision(process_label(test_set[1]), convert2preds(
        #     model.predict([x_test, tree_test])))[0]
        recall5 = code_level_accuracy(process_label(test_set[1]),convert2preds(
            model.predict([x_test, tree_test])))[0]
        # self.Precision_5.append(precision5)
        self.Recall_5.append(recall5)
        # metricsInfo = 'Epoch: %d, - Recall@5: %f, - Precision@5: %f' % (epoch+1, recall5, precision5)
        metricsInfo = 'Epoch: %d, - Recall@5: %f' % (epoch + 1, recall5)
        # print2file(metricsInfo, self.path, self.fileName)
        print(metricsInfo)

    def on_train_end(self, logs={}):
        print('Recall@5为:',self.Recall_5,'\n')
        print('Precision@5为:',self.Precision_5)
        # print2file('Recall@5:'+str(self.Recall_5), self.path, self.fileName)
        # print2file('Precision@5:'+str(self.Precision_5), self.path, self.fileName)


def print2file(buf, dirs, fileName):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    outFile = dirs + fileName
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()


if __name__ == '__main__':
    seqFile = '../resource/mimic3.seqs'
    knowledgeFile = '../resource/mimic3_newTree.seqs'
    labelFile = '../resource/mimic3.allLabels'
    gcn_emb = pickle.load(open('../resource/gcn_emb_onehot.emb', 'rb'))

    # node2vec Embedding
    diagcode_emb = np.load('../resource/node2vec_emb/diagcode_emb.npy')
    # knowledge_emb = np.load('../resource/node2vec_emb/knowledge_emb.npy')

    # gcn Embedding
    # diagcode_emb = gcn_emb[0][:4880]
    # knowledge_emb = gcn_emb[0][4880:]

    train_set, valid_set, test_set = load_data(seqFile, labelFile, knowledgeFile)
    x, y, tree = padMatrix(train_set[0], train_set[1],train_set[2])
    x_valid, y_valid, tree_valid = padMatrix(valid_set[0], valid_set[1], valid_set[2])
    x_test, y_test, tree_test = padMatrix(test_set[0], test_set[1], test_set[2])

    # test
    sta = np.random.random((x.shape[0], x.shape[1], 4))
    sta_valid = np.random.random((x_valid.shape[0], x_valid.shape[1], 4))
    K = np.random.random((x.shape[0], 18, 128))
    K_valid = np.random.random((x_valid.shape[0], 18, 128))
    V = np.random.random((x.shape[0], 18, 128))
    V_valid = np.random.random((x_valid.shape[0], 18, 128))

    model_input = keras.layers.Input((x.shape[1], x.shape[2]), name='model_input')
    K_input = keras.layers.Input((K.shape[1], K.shape[2]), name='K_input')
    V_input = keras.layers.Input((V.shape[1], V.shape[2]), name='V_input')
    sta_input = keras.layers.Input((sta.shape[1], sta.shape[2]), name='sta_input')
    mask = keras.layers.Masking(mask_value=0)(model_input)
    emb = keras.layers.Dense(128, activation='relu', kernel_initializer=keras.initializers.constant(diagcode_emb), name='emb')(mask)
    rnn = keras.layers.GRU(gru_dimentions, return_sequences=True, dropout=0.5)(emb)

    memory = memoryAttention(units=128, v_dimention=128)([K_input, rnn, V_input])
    q = coAttention(units=128, sta_dimention=128)([K_input, sta_input, memory, rnn])

    concat = keras.layers.concatenate([rnn, memory, q])

    model_output = keras.layers.Dense(labelCount, activation='softmax', name='main_output')(concat)

    model = keras.models.Model(inputs=[model_input, K_input, V_input, sta_input], outputs=model_output)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy')

    callback_history = metricsHistory()
    history = model.fit([x, K, V, sta], y,
                        epochs=1,
                        batch_size=10,
                        validation_data=([x_valid, K_valid, V_valid,sta_valid], y_valid))
