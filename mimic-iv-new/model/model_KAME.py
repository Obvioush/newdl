import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback
import _pickle as pickle
import numpy as np
import heapq
import operator
import os


_TEST_RATIO = 0.15
_VALIDATION_RATIO = 0.1
gru_dimentions = 128
codeCount = 8259  # icd9数
labelCount = 283  # 标签的类别数
treeCount = 728  # 分类树的祖先节点数量
timeStep = 145
train_epoch = 50
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


class KAMEAttention(keras.Model):
    def __init__(self, units):
        super(KAMEAttention, self).__init__()
        self.W1 = keras.layers.Dense(units)
        # self.W2 = keras.layers.Dense(units)
        # self.V = keras.layers.Dense(1)

    # tree_onehot, rnn_ht
    def call(self, inputs):
        Lt, rnn_ht = inputs
        # Lt.shape:(batch_size, length, size, units)=>(5250,41,84,128)
        # rnn_ht.shape:(batch_size, length, units) =>(5250,41,128)

        # ht.shape:(batch_size, length, size, units)=>(5250,41,1,128)
        ht = tf.expand_dims(rnn_ht, 2)
        Lt = self.W1(Lt)
        # score.shape:(batch_size, length, size, 1)
        score = tf.multiply(ht, Lt)

        # (Alpha)attention_weights.shape:(batch_size, length, size, 1)
        attention_weights = tf.nn.softmax(score, axis=2)

        # context_vector.shape: (batch_size, length, size, units)
        # tensorflow自己做扩展，把Alpha的1扩展为units
        context_vector = attention_weights * Lt

        # 在size维度求和，context_vector.shape:(batch_size, length, units)
        context_vector = tf.reduce_sum(context_vector, axis=2)

        return context_vector


def kame_knowledgematrix(treeseqs):
    # 和患者输入保持一致，访问为1到n-1
    for i in range(len(treeseqs)):
        treeseqs[i] = treeseqs[i][:-1]

    zerovec = np.zeros((87, 728), dtype=np.int8)
    ts = []
    for i in treeseqs:
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

    return ts


def treetonumpy(treeseqs):
    n_samples = len(treeseqs)

    tree = np.zeros((n_samples, timeStep, 87, treeCount), dtype=np.int8)
    for idx, tseq in enumerate(treeseqs):
        for tvec, subseq in zip(tree[idx, :, :, :], tseq[:-1]):
            # tvec[subseq] = 1.
            h1 = tvec
            h2 = subseq
    return tree


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


# class metricsHistory(Callback):
#     def __init__(self):
#         super().__init__()
#         self.Recall_5 = []
#         self.Precision_5 = []
#         self.path = 'G:\\mimic4_model_save\\model_KAME\\KAME_' + str(gru_dimentions)
#         # self.path = 'G:\\mimic4_model_save\\model_KAME\\KAME_' + str(gru_dimentions) + '_dropout02'
#         self.fileName = 'model_metrics.txt'
#         self.bestRecall = 0
#
#     def on_epoch_end(self, epoch, logs={}):
#         precision5 = visit_level_precision(process_label(test_set[1]), convert2preds(
#             model.predict([x_test, tree_test])))[0]
#         recall5 = code_level_accuracy(process_label(test_set[1]),convert2preds(
#             model.predict([x_test, tree_test])))[0]
#         self.Precision_5.append(precision5)
#         self.Recall_5.append(recall5)
#         # metricsInfo = 'Epoch: %d, - Recall@5: %f, - Precision@5: %f' % (epoch+1, recall5, precision5)
#         metricsInfo = 'Epoch: %d, - Recall@5: %f' % (epoch + 1, recall5)
#         if self.bestRecall < recall5:
#             self.bestRecall = recall5
#             if not os.path.exists(self.path):
#                 os.makedirs(self.path)
#             # model.save(self.path+'\\NKAM.' + str((epoch+1)) + '.h5')
#             tf.keras.models.save_model(model, self.path+'\\KAME_epoch_' + str((epoch+1)))
#
#         print2file(metricsInfo, self.path+'\\', self.fileName)
#         print(metricsInfo)
#
#     def on_train_end(self, logs={}):
#         print('Recall@5为:', self.Recall_5,'\n')
#         print('Precision@5为:', self.Precision_5)
#         print2file('Recall@5:'+str(self.Recall_5), self.path+'\\', self.fileName)
#         print2file('Precision@5:'+str(self.Precision_5), self.path+'\\', self.fileName)


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

    gloveKnowledgeFile = '../resource/gram_emb/mimic4_glove_knowledge_emb.npy'
    gramembFile = '../resource/gram_emb/gramemb_diagcode.npy'

    glove_knowledge_emb = np.load(gloveKnowledgeFile).astype(np.float32)
    gram_emb = np.load(gramembFile).astype(np.float32)

    # 测试tree的序列
    # tree_seq = pickle.load(open(treeFile, 'rb'))

    train_set, valid_set, test_set = load_data(seqFile, labelFile, treeFile)
    x, y = padMatrix(train_set[0], train_set[1])
    x_valid, y_valid = padMatrix(valid_set[0], valid_set[1])
    x_test, y_test = padMatrix(test_set[0], test_set[1])

    # KAME knowledge embedding
    tree = kame_knowledgematrix(train_set[2])
    tree = tf.convert_to_tensor(tree)
    # tree = treetonumpy(train_set[2])
    # tree_valid = kame_knowledgematrix(valid_set[2])
    # tree_valid = treetonumpy(tree_valid)
    # tree_test = kame_knowledgematrix(test_set[2], glove_knowledge_emb)

    # gram patient embedding
    # x = tf.matmul(x, tf.expand_dims(gram_emb, 0))
    # x_valid = tf.matmul(x_valid, tf.expand_dims(gram_emb, 0))
    # x_test = tf.matmul(x_test, tf.expand_dims(gram_emb, 0))


    gru_input = keras.layers.Input((x.shape[1], x.shape[2]), name='gru_input')
    mask = keras.layers.Masking(mask_value=0)(gru_input)
    v = keras.layers.Dense(128, activation='tanh', kernel_initializer=keras.initializers.constant(gram_emb))(mask)
    gru_out = keras.layers.GRU(gru_dimentions, return_sequences=True, dropout=0.5)(v)

    tree_input = keras.layers.Input((tree.shape[1], tree.shape[2], tree.shape[3]), name='tree_input')
    mask1 = keras.layers.Masking(mask_value=0)(tree_input)
    tree_emb = keras.layers.Dense(128, kernel_initializer=keras.initializers.constant(glove_knowledge_emb))(mask1)
    context_vector = KAMEAttention(units=gru_dimentions)([tree_emb, gru_out])
    s = keras.layers.concatenate([gru_out, context_vector], axis=-1)
    main_output = keras.layers.Dense(labelCount, activation='softmax', name='main_output')(s)

    model = keras.models.Model(inputs=[gru_input, tree_input], outputs=main_output)

    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='G:\\mimic4_model_save\\model_KAME\\KAME_' + str(gru_dimentions) + '\\KAME_epoch_{epoch:02d}',
        monitor='val_loss',
        save_best_only=True,
        mode='auto')

    # callback_history = metricsHistory()
    callback_lists = [checkpoint]
    model.compile(optimizer='adam', loss='binary_crossentropy')

    history = model.fit([x, tree], y,
                        epochs=train_epoch,
                        batch_size=train_batch_size,
                        validation_data=([x_valid, tree_valid], y_valid),
                        callbacks=callback_lists)
