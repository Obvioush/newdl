import pickle
import numpy as np

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
    outFile = '../resource/mimic4'
    multidx = '../resource/ccs_multi_dx_tool_2015.csv'
    seqs = pickle.load(open('../resource/mimic4.seqs', 'rb'))
    types = pickle.load(open('../resource/mimic4.types', 'rb'))
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

    # types中0-8258为icd9编码
    category = {}
    for i in range(8259):
        category[i] = ref[retype[i]]

    for i in range(8259, len(types)):
        category[i] = convert_num(retype[i])

    c1 = []
    for k in category.items():
        c1.append(k)
    labels = np.array(c1)
    # idx_features_labels = np.loadtxt("{}{}.content".format("../model/data/cora/", "cora"), dtype=np.dtype(str))
    # labels = idx_features_labels[:, -1]
    classes = set(labels[:,-1])
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels[:,-1])), dtype=np.int32)
    # np.save('../resource/graphLabel', labels_onehot)
