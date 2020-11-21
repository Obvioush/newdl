import pickle


def convert_to_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3] + '.' + dxStr[3:]
        else: return dxStr


if __name__ == '__main__':
    outFile = '../resource/mimic3'
    dxref = '../resource/$dxref 2015.csv'
    seqs = pickle.load(open('../resource/mimic3.seqs', 'rb'))
    types = pickle.load(open('../resource/mimic3.types', 'rb'))
    retype = dict(sorted([(v, k) for k, v in types.items()]))

    # 将单级分类中的icd-9编码按照ccs分组
    ref = {}
    infd = open(dxref, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().replace('\'', '').split(',')
        icd9 = 'D_' + convert_to_icd9(tokens[0].replace(' ', ''))
        ccs = int(tokens[1].replace(' ', ''))
        ref[icd9] = ccs
    infd.close()

    temp = {}  # temp的key为icd9转化后的编码(0-4879), value为ccs的分类编码
    category = []
    for k, v in retype.items():
        if k == 4880: break
        temp[k] = ref[v]
        if ref[v] not in category:
            category.append(ref[v])

    length = len(category)

    ccsMap = {}  # MIMIC3的数据中一共有272个唯一ccs分类编码, key:ccs分类编码, value:0-271
    for i in range(len(category)):
        ccsMap[category[i]] = i

    indexMap = {}  # key:icd9的index(0-4879), value: ccs唯一分组(0-271)
    for k, v in temp.items():
        indexMap[k] = ccsMap[v]

    newLabel = []
    for patient in seqs:
        newVisit = []
        for visit in patient[-1]:
            if indexMap[visit] not in newVisit:
                newVisit.append(indexMap[visit])
        newLabel.append(newVisit)

    # pickle.dump(newLabel, open(outFile + '.labels', 'wb'), -1)
