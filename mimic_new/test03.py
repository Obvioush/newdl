import tensorflow as tf
import tensorflow.keras as keras
import _pickle as pickle
import numpy as np
import heapq
import operator


_TEST_RATIO = 0.2
_VALIDATION_RATIO = 0.1
gru_dimentions = 128


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


def padMatrix(seqs, labels):
    lengths = np.array([len(seq) for seq in seqs]) - 1
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    inputDimSize = calculate_dimSize('./resource/process_data/process.dataseqs')
    numClass = calculate_dimSize('./resource/process_data/process.labelseqs')

    x = np.zeros((maxlen, n_samples, inputDimSize)).astype(np.float32)
    y = np.zeros((maxlen, n_samples, numClass)).astype(np.float32)
    mask = np.zeros((maxlen, n_samples)).astype(np.float32)

    for idx, (seq, lseq) in enumerate(zip(seqs,labels)):
        for xvec, subseq in zip(x[:,idx,:], seq[:-1]):
            xvec[subseq] = 1.
        for yvec, subseq in zip(y[:,idx,:], lseq[1:]):
            yvec[subseq] = 1.
        mask[:lengths[idx], idx] = 1.

    lengths = np.array(lengths, dtype=np.float32)

    return x, y, mask, lengths


def padMatrix1(seqs, labels, treeseqs):
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


def get_random_weight(dim1, dim2, left=-0.1, right=0.1):
    return np.random.uniform(left, right, (dim1, dim2)).astype(np.float32)

@tf.function
def generate_latentMatrix(treesonehot):
    x = []
    testA = get_random_weight(gru_dimentions, 729)

    for patient in treesonehot:
        newPatient = []
        for visit in patient:
            newVisit = []
            for index, code in enumerate(visit):
                if code == 1:
                    newVisit.append(testA[:, index])
            if newVisit:
                newPatient.append(np.array(newVisit))
            else:
                newPatient.append(newPatient[-1])
        x.append(np.array(newPatient))

    # for patient in treesonehot:
    #     newPatient = []
    #     for visit in patient:
    #         newVisit = []
    #         for code in visit:
    #             newVisit.append(testA[:, code])
    #         newPatient.append(newVisit)
    #     if len(patient) < 41:
    #          #
    #     x.append(newPatient)
    return x


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


# class TreeEncoder(keras.Model):
#     def __init__(self, vocab_size, embedding_units, encoding_units,
#                  batch_size):
#         super(TreeEncoder, self).__init__()
#         self.batch_size = batch_size
#         self.encoding_units = encoding_units
#         self.embedding = keras.layers.Embedding(vocab_size, embedding_units)
#         self.V = keras.layers.Dense(1)
#         self.temp = np.zeros(729)
#         self.W = self.add_weight(name='att_weight',
#                                  shape=(200, 729),
#                                  initializer='uniform',
#                                  trainable=True)
#
#     def call(self, x, hidden):
#         x = self.embedding(x)
#
#
#
#         output, state = self.gru(x, initial_state=hidden)
#         return output, state


# def Knowledgeattention(knowledge_onehot, encoder_outputs):
#     # latent_knowledge_matrix.shape:(batch_size, length, ancestorNum, units)
#     # encoder_outputs.shape(batch_size, length, units)
#     latent_knowledge_matrix = generate_latentMatrix(knowledge_onehot)
#     encoder_outputs_with_ancestorNum_axis = tf.expand_dims(encoder_outputs, 2)
#     score = keras.layers.dot(latent_knowledge_matrix, encoder_outputs_with_ancestorNum_axis)
#     # shape: (batch_size, length, ancestorNum, 1)
#     attention_weights = tf.nn.softmax(score, axis=1)
#     # context_vector.shape: (batch_size, length, ancestorNum, units)
#     context_vector = attention_weights * latent_knowledge_matrix
#     # context_vector.shape: (batch_size, length, units)
#     context_vector = tf.reduce_sum(context_vector, axis=1)
#
#     return context_vector, attention_weights

class KnowledgeAttention(keras.Model):
    def __init__(self, units):
        super(KnowledgeAttention, self).__init__()
        self.W1 = keras.layers.Dense(units)
        self.W2 = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)


    def call(self, knowledge_onehot, encoder_outputs):
        # decoder_hidden.shape:(batch_size, length, units)
        # encoder_outputs.shape(batch_size, length, units)

        # before V: (batch_size, length, units)
        # after V: (batch_size, length, 1)
        context_vector_all = None
        for i in range(encoder_outputs.shape[1]):
            encoder_output = tf.expand_dims(encoder_outputs[:,i,:],1)
            score = self.V(tf.nn.tanh(
                self.W1(encoder_output) + self.W2(knowledge_onehot)))
            # shape: (batch_size, length, 1)
            attention_weights = tf.nn.softmax(score, axis=1)
            # context_vector.shape: (batch_size, length, units)
            context_vector = attention_weights * knowledge_onehot
            # context_vector.shape: (batch_size, units)
            context_vector = tf.reduce_sum(context_vector, axis=1)
            context_vector = tf.expand_dims(context_vector,1)
            if context_vector_all is None:
                context_vector_all = context_vector
            else:
                context_vector_all = keras.layers.concatenate([context_vector_all,context_vector],axis=1)

        return context_vector_all


if __name__ == '__main__':
    seqFile = './resource/process_data/process.dataseqs'
    labelFile = './resource/process_data/process.labelseqs'
    treeFile = './resource/process_data/process_new.treeseqs'
    # data_seqs = pickle.load(open('./resource/process_data/process.dataseqs', 'rb'))
    # label_seqs = pickle.load(open('./resource/process_data/process.labelseqs', 'rb'))
    # types = pickle.load(open('./resource/build_trees.types', 'rb'))
    # retype = dict([(v, k) for k, v in types.items()])

    train_set, valid_set, test_set = load_data(seqFile, labelFile, treeFile)
    x, y, tree, lengths = padMatrix1(train_set[0], train_set[1], train_set[2])
    x_valid, y_valid, tree_valid, valid_lengths = padMatrix1(valid_set[0], valid_set[1], valid_set[2])
    x_test, y_test, tree_test, test_lengths = padMatrix1(test_set[0], test_set[1], test_set[2])
    # tree1 = np.array(train_set[2]).astype(np.int32)

    # model = keras.models.Sequential([
    #     # 添加一个Masking层，这个层的input_shape=(timesteps, features)
    #     keras.layers.Masking(mask_value=0, input_shape=(x.shape[1], x.shape[2])),
    #     # keras.layers.LSTM(64, return_sequences=True),
    #     # keras.layers.SimpleRNN(1800, return_sequences=True),
    #     # keras.layers.GRU(64, dropout=0.2, return_sequences=True, unroll=True),
    #     keras.layers.SimpleRNN(64, return_sequences=True),
    #     # keras.layers.TimeDistributed(keras.layers.Dense(283, activation='softmax'))
    #     keras.layers.Dense(283, activation='softmax')
    #
    # ])

    gru_input = keras.layers.Input((x.shape[1], x.shape[2]), name='gru_input')
    mask = keras.layers.Masking(mask_value=0)(gru_input)
    gru_out = keras.layers.GRU(gru_dimentions, return_sequences=True)(mask)

    tree_input = keras.layers.Input((tree.shape[1], tree.shape[2]), name='tree_input')
    mask1 = keras.layers.Masking(mask_value=0)(tree_input)
    mask1 = keras.layers.Dense(gru_dimentions)(mask1)
    ka = KnowledgeAttention(units=128)
    context_vector = ka(mask1, gru_out)
    # knowledge_vector = tf.tile(tf.expand_dims(context_vector, 1), [1, x.shape[1], 1])
    # s = keras.layers.concatenate([gru_out, knowledge_vector], axis=-1)
    s = keras.layers.concatenate([gru_out, context_vector], axis=-1)

    main_output = keras.layers.TimeDistributed(keras.layers.Dense(283, activation='softmax'))(s)
    # main_output = keras.layers.Dense(283, activation='softmax', name='main_output')(s)

    model = keras.models.Model(inputs=[gru_input, tree_input], outputs=main_output)

    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # history = model.fit([x, tree], y,
    #                     epochs=2,
    #                     batch_size=100,
    #                     validation_data=([x_valid, tree_valid], y_valid))
    #
    # preds = model.predict([x_test, tree_test], batch_size=100)
    #
    # # def recallTop(y_true, y_pred, rank=[10, 20, 30]):
    # #     recall = list()
    # #     for i in range(len(y_pred)):
    # #         thisOne = list()
    # #         codes = y_true[i]
    # #         tops = y_pred[i]
    # #         for rk in rank:
    # #             thisOne.append(len(set(codes).intersection(set(tops[:rk])))*1.0/len(set(codes)))
    # #         recall.append(thisOne)
    # #     return (np.array(recall)).mean(axis=0).tolist()
    #
    # def visit_level_precision(y_true, y_pred, rank=[10, 20, 30]):
    #     recall = list()
    #     for i in range(len(y_true)):
    #         for j in range(len(y_true[i])):
    #             thisOne = list()
    #             codes = y_true[i][j]
    #             tops = y_pred[i][j]
    #             for rk in rank:
    #                 thisOne.append(len(set(codes).intersection(set(tops[:rk]))) * 1.0 / min(rk, len(set(codes))))
    #             recall.append(thisOne)
    #     return (np.array(recall)).mean(axis=0).tolist()
    #
    #
    # def codel_level_accuracy(y_true, y_pred, rank=[10, 20, 30]):
    #     recall = list()
    #     for i in range(len(y_true)):
    #         for j in range(len(y_true[i])):
    #             thisOne = list()
    #             codes = y_true[i][j]
    #             tops = y_pred[i][j]
    #             for rk in rank:
    #                 thisOne.append(len(set(codes).intersection(set(tops[:rk]))) * 1.0 / len(set(codes)))
    #             recall.append(thisOne)
    #     return (np.array(recall)).mean(axis=0).tolist()
    #
    #
    # # 按从大到小取预测值中前30个ccs分组号
    # def convert2preds(preds):
    #     ccs_preds = []
    #     for i in range(len(preds)):
    #         temp = []
    #         for j in range(len(preds[i])):
    #             temp.append(list(zip(*heapq.nlargest(30, enumerate(preds[i][j]), key=operator.itemgetter(1))))[0])
    #         ccs_preds.append(temp)
    #     return ccs_preds
    #
    # y_pred = convert2preds(preds)
    # y_true = process_label(test_set[1])
    # metrics_visit_level_precision = visit_level_precision(y_true, y_pred)
    # metrics_codel_level_accuracy = codel_level_accuracy(y_true, y_pred)
    #
    # print("Top-10 visit_level_precision为：", metrics_visit_level_precision[0])
    # print("Top-20 visit_level_precision为：", metrics_visit_level_precision[1])
    # print("Top-30 visit_level_precision为：", metrics_visit_level_precision[2])
    # print("---------------------------------------------------------")
    # print("Top-10 codel_level_accuracy为：", metrics_codel_level_accuracy[0])
    # print("Top-20 codel_level_accuracy为：", metrics_codel_level_accuracy[1])
    # print("Top-30 codel_level_accuracy为：", metrics_codel_level_accuracy[2])
