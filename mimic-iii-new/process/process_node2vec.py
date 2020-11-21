import pickle
import numpy as np


# node2vec训练的词向量
embedding_index = {}
emb_matrix = []

with open('../resource/node2vec_emb/node2vec_mimic3_emb.txt', encoding='UTF-8') as f:
    f.readline()
    for line in f:
        tokens = line.split()
        node = int(tokens[0])
        emb_values = np.asarray(tokens[1:], dtype=np.float32)
        embedding_index[node] = emb_values

for i in range(len(embedding_index)):
    emb_matrix.append(embedding_index[i])

diagcode_emb = np.asarray(emb_matrix[:4880], dtype=np.float32)
knowledge_emb = np.asarray(emb_matrix[4880:], dtype=np.float32)
patient_emb = np.asarray(emb_matrix, dtype=np.float32)


# np.save('../resource/node2vec_emb/diagcode_emb', diagcode_emb)
# np.save('../resource/node2vec_emb/knowledge_emb', knowledge_emb)
# np.save('../resource/node2vec_emb/patient_emb', patient_emb)
