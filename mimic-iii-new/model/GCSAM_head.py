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

        QK = QK / (128 ** 0.5)

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
    # diagcode_emb = np.load('../resource/node2vec_emb/diagcode_emb.npy')
    # knowledge_emb = np.load('../resource/node2vec_emb/knowledge_emb.npy')

    # gcn Embedding
    diagcode_emb = gcn_emb[0][:4880]
    knowledge_emb = gcn_emb[0][4880:]

    train_set, valid_set, test_set = load_data(seqFile, labelFile, knowledgeFile)
    x, y, tree = padMatrix(train_set[0], train_set[1],train_set[2])
    x_valid, y_valid, tree_valid = padMatrix(valid_set[0], valid_set[1], valid_set[2])
    x_test, y_test, tree_test = padMatrix(test_set[0], test_set[1], test_set[2])

    model_input = keras.layers.Input((x.shape[1], x.shape[2]), name='model_input')
    mask = keras.layers.Masking(mask_value=0)(model_input)
    emb = keras.layers.Dense(128, activation='relu', kernel_initializer=keras.initializers.constant(diagcode_emb), name='emb')(mask)
    rnn = keras.layers.Bidirectional(keras.layers.GRU(gru_dimentions, return_sequences=True, dropout=0.5))(emb)
    head1 = keras.layers.Attention()([rnn, rnn])
    head2 = keras.layers.Attention()([rnn, rnn])
    head3 = keras.layers.Attention()([rnn, rnn])
    head4 = keras.layers.Attention()([rnn, rnn])
    head5 = keras.layers.Attention()([rnn, rnn])
    head6 = keras.layers.Attention()([rnn, rnn])
    head7 = keras.layers.Attention()([rnn, rnn])
    head8 = keras.layers.Attention()([rnn, rnn])

    tree_input = keras.layers.Input((tree.shape[1], tree.shape[2]), name='tree_input')
    tree_mask = keras.layers.Masking(mask_value=0)(tree_input)
    tree_emb = keras.layers.Dense(128, kernel_initializer=keras.initializers.constant(knowledge_emb),
                                    trainable=True, name='tree_emb')(tree_mask)

    head1, weights1 = ScaledDotProductAttention(output_dim=128)([tree_emb, head1])
    head2, weights2 = ScaledDotProductAttention(output_dim=128)([tree_emb, head2])
    head3, weights3 = ScaledDotProductAttention(output_dim=128)([tree_emb, head3])
    head4, weights4 = ScaledDotProductAttention(output_dim=128)([tree_emb, head4])
    head5, weights5 = ScaledDotProductAttention(output_dim=128)([tree_emb, head5])
    head6, weights6 = ScaledDotProductAttention(output_dim=128)([tree_emb, head6])
    head7, weights7 = ScaledDotProductAttention(output_dim=128)([tree_emb, head7])
    head8, weights8 = ScaledDotProductAttention(output_dim=128)([tree_emb, head8])

    st = keras.layers.concatenate([head1, head2, head3,head4, head5, head6, head7], axis=-1)
    model_output = keras.layers.Dense(labelCount, activation='softmax', name='main_output')(st)
    # model_output = keras.layers.Dense(labelCount, activation='softmax', name='main_output')(head1)

    model = keras.models.Model(inputs=[model_input, tree_input], outputs=model_output)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001, decay=0.001), loss='binary_crossentropy')

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='G:\\mimic3_model_save\\model_GCSAM\\test_head\\head_8\\NEW_epoch_{epoch:02d}',
        monitor='val_loss',
        save_best_only=True,
        mode='auto')
    callback_history = [checkpoint]
    history = model.fit([x, tree], y,
                        epochs=100,
                        batch_size=100,
                        validation_data=([x_valid, tree_valid], y_valid),
                        callbacks=callback_history)
