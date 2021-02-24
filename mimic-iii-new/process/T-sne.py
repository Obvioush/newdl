import pickle
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

def convert_to_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3] + '.' + dxStr[3:]
        else: return dxStr

def convert_num(dxStr):
    if len(dxStr) == 1: return int(dxStr[:])
    else :
        if dxStr[:2]=='A_': return int(0)
        return int(dxStr[:2].replace('.', ''))


if __name__ == '__main__':
    outFile = '../resource/mimic3'
    multidx = '../resource/ccs_multi_dx_tool_2015.csv'
    seqs = pickle.load(open('../resource/mimic3.seqs', 'rb'))
    types = pickle.load(open('../resource/mimic3.types', 'rb'))
    retype = dict(sorted([(v, k) for k, v in types.items()]))

    # 将多级分类中的icd-9编码按照多级ccs分组
    ref = {}
    infd = open(multidx, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().replace('\'', '').split(',')
        icd9 = 'D_' + convert_to_icd9(tokens[0].replace(' ', ''))
        multiccs = int(tokens[1].replace(' ', ''))
        ref[icd9] = multiccs
    infd.close()

    # types中0-4879为icd9编码
    category = {}
    for i in range(4880):
        category[i] = ref[retype[i]]

    for i in range(4880,len(types)):
        category[i] = convert_num(retype[i])

    c1 = []
    for k in category.items():
        c1.append(k)
    labels = np.array(c1)
    # idx_features_labels = np.loadtxt("{}{}.content".format("../model/data/cora/", "cora"), dtype=np.dtype(str))
    # labels = idx_features_labels[:, -1]
    # classes = set(labels[:,-1])
    # classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    # labels_onehot = np.array(list(map(classes_dict.get, labels[:,-1])), dtype=np.int32)
    # np.save('../resource/graphLabel', labels_onehot)

    label_tsne = labels[:4880, 1]
    # np.save('../resource/tsne_label', label_tsne)

    gcn_emb_load = pickle.load(open('../resource/gcn_emb_onehot.emb', 'rb'))

    # gcn Embedding
    gcn_emb = gcn_emb_load[0][:4880]

    # glove Embedding
    glove_emb = np.load('../resource/gram_emb/mimic3_glove_patient_emb.npy')

    # gram_emb = np.load('../resource/gram_emb/gramemb_diagcode.npy')

    model = TSNE()
    np.set_printoptions(suppress=True)
    Y1 = model.fit_transform(gcn_emb)  # 将X降维(默认二维)后保存到Y中
    Y2 = model.fit_transform(glove_emb)  # 将X降维(默认二维)后保存到Y中
    # Y3 = model.fit_transform(gram_emb)

    fig = plt.figure(dpi=600)
    ax1 = fig.add_subplot(111)
    ax1.scatter(Y1[:, 0], Y1[:, 1], 1, label_tsne)  # labels为每一行对应标签，20为标记大小
    # ax1.scatter(Y2[:, 0], Y2[:, 1], 2, label_tsne)  # labels为每一行对应标签，20为标记大小
    plt.axis('off')
    ax1.axis('off')
    # plt.savefig("./gcn_emb.eps", dpi=600)  # 保存图片
    plt.show()