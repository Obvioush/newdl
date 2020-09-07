import _pickle as pickle
import numpy as np


# def load_embedding(options):
#     m = np.load(options)
#     w = (m['w'] + m['w_tilde']) / 2.0
#     return w


if __name__ == '__main__':
    # alltype = pickle.load(open('/Users/masaka/Documents/mimic相关/process_mimic数据/node2vec_edgelist.oldtypes', 'rb'))
    # icd_type = pickle.load(open('./resource/build_trees.types', 'rb'))
    # retype = dict([(v, k) for k, v in alltype.items()])
    # icd_retype = dict([(v, k) for k, v in icd_type.items()])
    #
    # aps = dict(sorted(retype.items(), key=lambda a:a[0]))
    # icd_aps = dict(sorted(icd_retype.items(), key=lambda a:a[0]))
    #
    # for i in range(len(aps)):
    #     if aps[i].startswith('D'):
    #         aps[i] = aps[i].strip('D_').replace('.', '')
    #     if aps[i].startswith('A'):
    #         aps[i] = aps[i].strip('A_')
    #
    # for i in range(len(icd_aps)):
    #     if icd_aps[i].startswith('D'):
    #         icd_aps[i] = icd_aps[i].strip('D_').replace('.', '')

    # node2vec训练的词向量
    # vocab_size = 728
    # embedding_dim = 128
    # embedding_index = {}
    #
    # with open('/Users/masaka/Documents/mimic相关/process_mimic数据/vec_all128.txt', encoding='UTF-8') as f:
    #     for line in f:
    #         tokens = line.split()
    #         emb_node = tokens[0]
    #         emb_values = np.asarray(tokens[1:], dtype='float32')
    #         embedding_index[emb_node] = emb_values
    #
    # knowledge_emb = np.zeros((vocab_size, embedding_dim))
    # for i in range(vocab_size):
    #     # knowledge元素从aps的4894开始，共728个
    #     temp = aps.get(i+4894)
    #     if temp is not None:
    #         knowledge_emb[i] = embedding_index[temp]
    # # np.save('/Users/masaka/Documents/mimic相关/process_mimic数据/comap/node2vec_test', knowledge_emb)
    #
    # patient_emb = np.zeros((4893, embedding_dim))
    # for i in range(4893):
    #     # knowledge元素从aps的4894开始，共728个
    #     temp = icd_aps.get(i)
    #     if temp is not None:
    #             patient_emb[i] = embedding_index[temp]
    # # np.save('/Users/masaka/Documents/mimic相关/process_mimic数据/comap/node2vec_patient_test', patient_emb)


    # glove embedding相关
    a = np.load('./resource/embedding/gram_128.33.npz')
    # b = a['W_emb'][:4893]
    # np.save('./resource/embedding/gram_128', b)


    # glove_emb = load_embedding('./resource/embedding/gram_128.33.npz')
    # glove_patient_emb = glove_emb[:4893]
    # glove_knowledge_emb = glove_emb[4894:]
    # np.save('/Users/masaka/Documents/mimic相关/process_mimic数据/comap/glove_test', glove_patient_emb)
    # np.save('/Users/masaka/Documents/mimic相关/process_mimic数据/comap/glove_knowledge_test', glove_knowledge_emb)
    # emb1 = np.load('/Users/masaka/Documents/mimic相关/process_mimic数据/comap/glove_test.npy')

