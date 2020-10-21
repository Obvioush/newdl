import _pickle as pickle
import numpy as np


def load_embedding(options):
    m = np.load(options)
    w = (m['w'] + m['w_tilde']) / 2.0
    return w


if __name__ == '__main__':
    # node2vec相关

    # alltype = pickle.load(open('./resource/mimic4.oldtypes', 'rb'))
    # icd_type = pickle.load(open('./resource/mimic4.types', 'rb'))
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
    #
    # # node2vec训练的词向量
    # vocab_size = 728
    # embedding_dim = 128
    # embedding_index = {}
    #
    # with open('./resource/node2vec_d128.txt', encoding='UTF-8') as f:
    #     for line in f:
    #         tokens = line.split()
    #         emb_node = tokens[0]
    #         emb_values = np.asarray(tokens[1:], dtype='float32')
    #         embedding_index[emb_node] = emb_values
    #
    # # 分类知识的node2vec嵌入
    # knowledge_emb = np.zeros((vocab_size, embedding_dim))
    # for i in range(vocab_size):
    #     # knowledge元素从aps的8259开始，共728个
    #     temp = aps.get(i+8259)
    #     if temp is not None:
    #         knowledge_emb[i] = embedding_index[temp]
    # np.save('./resource/process_data/mimic4_node2vec_d128', knowledge_emb)

    # # 患者就诊的node2vec嵌入
    # patient_emb = np.zeros((4893, embedding_dim))
    # for i in range(4893):
    #     # knowledge元素从aps的4894开始，共728个
    #     temp = icd_aps.get(i)
    #     if temp is not None:
    #             patient_emb[i] = embedding_index[temp]
    # # np.save('/Users/masaka/Documents/mimic相关/process_mimic数据/comap/node2vec_patient_test', patient_emb)


    # glove embedding相关

    # a = np.load('./resource/embedding/gram_128.33.npz')
    # b = a['W_emb'][:4893]
    # np.save('./resource/embedding/gram_128', b)

    glove_emb = load_embedding('./resource/embedding/comap.63.npz')
    # np.save('./resource/embedding/gram_glove_all', glove_emb)
    glove_patient_emb = glove_emb[:4893]
    glove_knowledge_emb = glove_emb[4894:]
    np.save('./resource/embedding/glove_patient', glove_patient_emb)
    np.save('./resource/embedding/glove_knowledge', glove_knowledge_emb)
    # emb1 = np.load('/Users/masaka/Documents/mimic相关/process_mimic数据/comap/glove_test.npy')

