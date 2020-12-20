import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import _pickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq
import operator
import os

_TEST_RATIO = 0.15
_VALIDATION_RATIO = 0.1
gru_dimentions = 256  # 384  512  640
codeCount = 6534  # icd9数
labelCount = 277  # 标签的类别数
treeCount = 728  # 分类树的祖先节点数量
timeStep = 145
train_epoch = 100
train_batch_size = 100

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


class MyEmbedding(keras.layers.Layer):
    def __init__(self,embedding_matrix, **kwargs):
        self.embedding_matrix = embedding_matrix
        super(MyEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        # assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='embedding_matrix',
                                      shape=(self.embedding_matrix.shape[0], self.embedding_matrix.shape[1]),
                                      initializer=keras.initializers.constant(self.embedding_matrix),
                                      trainable=True)
        super(MyEmbedding, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        # assert isinstance(x, list)
        emb = K.tanh(K.dot(inputs, self.kernel))
        return emb


class ScaledDotProductAttention(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.output_dim = output_dim
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(2, input_shape[0][2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[1][2], self.output_dim),
                                 initializer='uniform',
                                 trainable=True)

        super(ScaledDotProductAttention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):
        Lt, rnn_ht = inputs
        WQ = K.dot(rnn_ht, self.W)
        WK = K.dot(Lt, self.kernel[0])
        WV = K.dot(Lt, self.kernel[1])
        # WQ.shape (None, 41, 128)
        # print("WQ.shape", WQ.shape)
        # 转置 K.permute_dimensions(WK, [0, 2, 1]).shape (None, 128, 41)
        # print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (64 ** 0.5)

        weights = K.softmax(QK)
        # QK.shape (None, 41, 41)
        # print("QK.shape", weights.shape)

        context_vector = K.batch_dot(weights, WV)

        return context_vector, weights

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
        })
        return config


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
        self.path = 'G:\\mimic4_small_model_save\\model_NKAM\\NKAM_' + str(gru_dimentions)
        # self.path = 'G:\\mimic4_small_model_save\\model_NKAM\\NKAM_' + str(gru_dimentions) + '_dropout08'
        self.fileName = 'model_metrics.txt'
        self.bestRecall = 0

    def on_epoch_end(self, epoch, logs={}):
        # precision5 = visit_level_precision(process_label(test_set[1]), convert2preds(
        #     model.predict([x_test, tree_test])))[0]
        recall5 = code_level_accuracy(process_label(test_set[1]),convert2preds(
            model.predict([x_test, tree_test])))[0]
        # self.Precision_5.append(precision5)
        # self.Recall_5.append(recall5)
        if self.bestRecall < recall5:
            self.bestRecall = recall5
        # metricsInfo = 'Epoch: %d, - Recall@5: %f, - Precision@5: %f' % (epoch+1, recall5, precision5)
        # metricsInfo = 'Epoch: %d, - Recall@5: %f' % (epoch + 1, recall5)
        metricsInfo = 'Epoch: %d, - Recall@5: %f, - best recall@5: %f' % (epoch + 1, recall5, self.bestRecall)
        # if self.bestRecall < recall5:
        #     self.bestRecall = recall5
        #     if not os.path.exists(self.path):
        #         os.makedirs(self.path)
        #     tf.keras.models.save_model(model, self.path+'\\NKAM_epoch_' + str((epoch+1)))

        print2file(metricsInfo, self.path+'\\', self.fileName)
        print(metricsInfo)

    def on_train_end(self, logs={}):
        print2file('best recall@5:'+str(self.bestRecall), self.path+'\\', self.fileName)
        # print('Recall@5为:', self.Recall_5,'\n')
        # print('Precision@5为:', self.Precision_5)
        # print2file('Recall@5:'+str(self.Recall_5), self.path+'\\', self.fileName)
        # print2file('Precision@5:'+str(self.Precision_5), self.path+'\\', self.fileName)


def print2file(buf, dirs, fileName):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    outFile = dirs + fileName
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()


if __name__ == '__main__':
    seqFile = '../resource/mimic4.seqs'
    labelFile = '../resource/mimic4.allLabels'
    treeFile = '../resource/mimic4_newTree.seqs'

    # node2vec Embedding
    diagcode_emb = np.load('../resource/node2vec_emb/diagcode_emb.npy')
    knowledge_emb = np.load('../resource/node2vec_emb/knowledge_emb.npy')

    # gram Embedding
    # diagcode_emb = np.load('../resource/gram_emb/gramemb_diagcode.npy')

    train_set, valid_set, test_set = load_data(seqFile, labelFile, treeFile)
    x, y, tree = padMatrix(train_set[0], train_set[1], train_set[2])
    x_valid, y_valid, tree_valid = padMatrix(valid_set[0], valid_set[1], valid_set[2])
    x_test, y_test, tree_test = padMatrix(test_set[0], test_set[1], test_set[2])

    gru_input = keras.layers.Input((x.shape[1], x.shape[2]), name='gru_input')
    mask = keras.layers.Masking(mask_value=0)(gru_input)
    emb = keras.layers.Dense(128, activation='relu', kernel_initializer=keras.initializers.constant(diagcode_emb), name='diagcode_emb')(mask)
    gru_out = keras.layers.GRU(gru_dimentions, return_sequences=True, dropout=0.5)(emb)

    tree_input = keras.layers.Input((tree.shape[1], tree.shape[2]), name='tree_input')
    tree_mask = keras.layers.Masking(mask_value=0)(tree_input)
    tree_emb = keras.layers.Dense(128, activation='relu', kernel_initializer=keras.initializers.constant(knowledge_emb), name='knowledge_emb')(tree_mask)

    context_vector, weights = ScaledDotProductAttention(output_dim=128)([tree_emb, gru_out])
    st = keras.layers.concatenate([gru_out, context_vector], axis=-1)

    main_output = keras.layers.Dense(labelCount, activation='softmax', name='main_output')(st)

    model = keras.models.Model(inputs=[gru_input, tree_input], outputs=main_output)

    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath='G:\\mimic4_small_model_save\\model_NKAM\\NKAM_' + str(gru_dimentions) + '\\NKAM_epoch_{epoch:02d}',
    #     monitor='val_loss',
    #     save_best_only=True,
    #     mode='auto')

    callback_history = metricsHistory()
    callback_lists = [callback_history]
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy')

    history = model.fit([x, tree], y,
                        epochs=train_epoch,
                        batch_size=train_batch_size,
                        validation_data=([x_valid, tree_valid], y_valid),
                        callbacks=callback_lists)
