import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tcn import TCN
import _pickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq
import operator
import os

_TEST_RATIO = 0.15
_VALIDATION_RATIO = 0.1
gru_dimentions = 128

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


def padMatrix(seqs, labels, treeseqs):
    lengths = np.array([len(seq) for seq in seqs]) - 1
    n_samples = len(seqs)
    # maxlen = np.max(lengths)
    maxlen = 41

    inputDimSize = calculate_dimSize('./resource/process_data/process.dataseqs')
    numClass = calculate_dimSize('./resource/process_data/process.labelseqs')
    treeDimSize = calculate_dimSize('./resource/process_data/process_new.treeseqs')

    x = np.zeros((n_samples, maxlen, inputDimSize)).astype(np.float32)
    y = np.zeros((n_samples, maxlen, numClass)).astype(np.float32)
    tree = np.zeros((n_samples, maxlen, treeDimSize)).astype(np.float32)
    # mask = np.zeros((maxlen, n_samples)).astype(np.float32)

    for idx, (seq, lseq, tseq) in enumerate(zip(seqs, labels, treeseqs)):
        for xvec, subseq in zip(x[idx, :, :], seq[:-1]):
            xvec[subseq] = 1.
        for yvec, subseq in zip(y[idx, :, :], lseq[1:]):
            yvec[subseq] = 1.
        for tvec, subseq in zip(tree[idx, :, :], tseq[:-1]):
            tvec[subseq] = 1.
        # mask[:lengths[idx], idx] = 1.

    lengths = np.array(lengths, dtype=np.float32)

    return x, y, tree, lengths


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


# class ScaledDotProductAttention(keras.layers.Layer):
#     def __init__(self, masking=True, future=False, dropout_rate=0., **kwargs):
#         self._masking = masking
#         self._future = future
#         self._dropout_rate = dropout_rate
#         self._masking_num = -2 ** 32 + 1
#         super(ScaledDotProductAttention, self).__init__(**kwargs)
#
#     def mask(self, inputs, masks):
#         masks = K.cast(masks, 'float32')
#         masks = K.tile(masks, [K.shape(inputs)[0] // K.shape(masks)[0], 1])
#         masks = K.expand_dims(masks, 1)
#         outputs = inputs + masks * self._masking_num
#         return outputs
#
#     def future_mask(self, inputs):
#         diag_vals = tf.ones_like(inputs[0, :, :])
#         tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
#         future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])
#         paddings = tf.ones_like(future_masks) * self._masking_num
#         outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
#         return outputs
#
#     def call(self, inputs):
#         if self._masking:
#             assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
#             queries, keys, values, masks = inputs
#         else:
#             assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
#             queries, keys, values = inputs
#
#         if K.dtype(queries) != 'float32':  queries = K.cast(queries, 'float32')
#         if K.dtype(keys) != 'float32':  keys = K.cast(keys, 'float32')
#         if K.dtype(values) != 'float32':  values = K.cast(values, 'float32')
#
#         matmul = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1]))  # MatMul
#         scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5  # Scale
#         if self._masking:
#             scaled_matmul = self.mask(scaled_matmul, masks)  # Mask(opt.)
#
#         if self._future:
#             scaled_matmul = self.future_mask(scaled_matmul)
#
#         softmax_out = K.softmax(scaled_matmul)  # SoftMax
#         # Dropout
#         out = K.dropout(softmax_out, self._dropout_rate)
#
#         outputs = K.batch_dot(out, values)
#
#         return outputs
#
#     def compute_output_shape(self, input_shape):
#         return input_shape


class MultiHeadAttention(keras.layers.Layer):

    def __init__(self, n_heads, head_dim, dropout_rate=.1, masking=False, future=False, trainable=True, **kwargs):
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._dropout_rate = dropout_rate
        self._masking = masking
        self._future = future
        self._trainable = trainable
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self._weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_queries')
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_keys')
        self._weights_values = self.add_weight(
            shape=(input_shape[2][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_values')
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs):
        if self._masking:
            assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs

        queries_linear = K.dot(queries, self._weights_queries)
        keys_linear = K.dot(keys, self._weights_keys)
        values_linear = K.dot(values, self._weights_values)

        queries_multi_heads = tf.concat(tf.split(queries_linear, self._n_heads, axis=2), axis=0)
        keys_multi_heads = tf.concat(tf.split(keys_linear, self._n_heads, axis=2), axis=0)
        values_multi_heads = tf.concat(tf.split(values_linear, self._n_heads, axis=2), axis=0)

        if self._masking:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads, masks]
        else:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads]

        attention = ScaledDotProductAttention(
            masking=False, future=self._future, dropout_rate=self._dropout_rate)
        att_out = attention(att_inputs)

        outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class selfAttention(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.output_dim = output_dim
        super(selfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[0][2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        # self.W = self.add_weight(name='W',
        #                          shape=(input_shape[1][2], self.output_dim),
        #                          initializer='uniform',
        #                          trainable=True)

        super(selfAttention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):
        x = inputs[0]
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
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
        # self.path = 'G:\\模型训练保存\\ourmodel_' + str(gru_dimentions) + '_dropout\\rate05_02\\'
        # self.fileName = 'model_metrics.txt'

    def on_epoch_end(self, epoch, logs={}):
        # precision5 = visit_level_precision(process_label(test_set[1]), convert2preds(
        #     model.predict([x_test, net_test])))[0]
        recall5 = code_level_accuracy(process_label(test_set[1]),convert2preds(
            model.predict([x_test, net_test])))[0]
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
    seqFile = './resource/process_data/process.dataseqs'
    labelFile = './resource/process_data/process.labelseqs'
    treeFile = './resource/process_data/process_new.treeseqs'
    # glovePatientFile = './resource/embedding/glove_patient_test.npy'
    glovePatientFile = './resource/embedding/gram_128.npy'
    gloveKnowledgeFile = './resource/embedding/glove_knowledge_test.npy'
    node2vecFile = './resource/embedding/node2vec_test.npy'
    node2vecPatientFile = './resource/embedding/node2vec_patient_test.npy'
    gramembFile = './resource/embedding/gram_emb_final.npy'
    # data_seqs = pickle.load(open('./resource/process_data/process.dataseqs', 'rb'))
    # label_seqs = pickle.load(open('./resource/process_data/process.labelseqs', 'rb'))
    # types = pickle.load(open('./resource/build_trees.types', 'rb'))
    # retype = dict([(v, k) for k, v in types.items()])

    glove_patient_emb = np.load(glovePatientFile).astype(np.float32)
    glove_knowledge_emb = np.load(gloveKnowledgeFile).astype(np.float32)
    node2vec_patient_emb = np.load(node2vecPatientFile).astype(np.float32)
    node2vec_emb = np.load(node2vecFile).astype(np.float32)
    gram_emb = np.load(gramembFile).astype(np.float32)

    train_set, valid_set, test_set = load_data(seqFile, labelFile, treeFile)
    x, y, net, lengths = padMatrix(train_set[0], train_set[1], train_set[2])
    x_valid, y_valid, net_valid, valid_lengths = padMatrix(valid_set[0], valid_set[1], valid_set[2])
    x_test, y_test, net_test, test_lengths = padMatrix(test_set[0], test_set[1], test_set[2])

    # glove patient embedding
    # x = tf.matmul(x, tf.expand_dims(glove_patient_emb, 0))
    # x_valid = tf.matmul(x_valid, tf.expand_dims(glove_patient_emb, 0))
    # x_test = tf.matmul(x_test, tf.expand_dims(glove_patient_emb, 0))

    # node2vec patient embedding
    # x = tf.tanh(tf.matmul(x, tf.expand_dims(node2vec_patient_emb, 0)))
    # x_valid = tf.tanh(tf.matmul(x_valid, tf.expand_dims(node2vec_patient_emb, 0)))
    # x_test = tf.tanh(tf.matmul(x_test, tf.expand_dims(node2vec_patient_emb, 0)))

    # glove knowledge embedding
    # tree = tf.matmul(tree, tf.expand_dims(glove_knowledge_emb, 0))
    # tree_valid = tf.matmul(tree_valid, tf.expand_dims(glove_knowledge_emb, 0))
    # tree_test = tf.matmul(tree_test, tf.expand_dims(glove_knowledge_emb, 0))

    # node2vec knowledge embedding
    net = tf.matmul(net, tf.expand_dims(node2vec_emb, 0))
    net_valid = tf.matmul(net_valid, tf.expand_dims(node2vec_emb, 0))
    net_test = tf.matmul(net_test, tf.expand_dims(node2vec_emb, 0))

    gru_input = keras.layers.Input((x.shape[1], x.shape[2]), name='gru_input')
    mask = keras.layers.Masking(mask_value=0)(gru_input)
    embLayer = MyEmbedding(gram_emb)
    emb = embLayer(mask)
    tcn = TCN(nb_filters=128, return_sequences=True, use_skip_connections=True)(emb)
    # gru_out = keras.layers.GRU(gru_dimentions, return_sequences=True, dropout=0.5)(emb)
    # sa_out = keras.layers.Attention()([gru_out,gru_out,gru_out])

    net_input = keras.layers.Input((net.shape[1], net.shape[2]), name='tree_input')
    net_mask = keras.layers.Masking(mask_value=0)(net_input)
    # net_embLayer = MyEmbedding(node2vec_emb)
    # net_emb = net_embLayer(net_mask)
    context_vector, weights = ScaledDotProductAttention(output_dim=128)([net_mask, tcn])
    st = keras.layers.concatenate([tcn, context_vector], axis=-1)

    main_output = keras.layers.Dense(283, activation='softmax', name='main_output')(st)

    model = keras.models.Model(inputs=[gru_input, net_input], outputs=main_output)

    # checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='G:\\模型训练保存\\ourmodel_' + str(gru_dimentions) + '_dropout\\rate05_02\\model_{epoch:02d}', save_freq='epoch')

    callback_history = metricsHistory()
    callback_lists = [callback_history]
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy')

    history = model.fit([x, net], y,
                        epochs=10,
                        batch_size=100,
                        validation_data=([x_valid, net_valid], y_valid),
                        callbacks=callback_lists)


