import _pickle as pickle


def convert_to_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3] + '.' + dxStr[3:]
        else: return dxStr


if __name__ == '__main__':
    outFile = './resource/process_data/ccs'

    dxlabel = './resource/dxlabel 2015.csv'
    dxref = './resource/$dxref 2015.csv'
    seqs = pickle.load(open('./resource/mimic4.seqs', 'rb'))
    types = pickle.load(open('./resource/mimic4.types', 'rb'))

    # ccs分组标签整理，共283个
    label_ccs = {}  # 把283个ccs分组存入
    infd = open(dxlabel, 'r')
    count = 0
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        ccs = int(tokens[0])
        label_ccs[ccs] = count
        count += 1
    infd.close()

    # 将单级分类中的icd-9编码按照ccs分组
    label_ref = {}
    infd = open(dxref, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().replace('\'', '').split(',')
        icd9 = 'D_' + convert_to_icd9(tokens[0].replace(' ', ''))
        ccs = int(tokens[1].replace(' ', ''))
        label_ref[icd9] = label_ccs[ccs]
    infd.close()

    # 整理生成ccs分类的标签序列
    retype = dict([(v, k) for k, v in types.items()])
    label_seqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if label_ref[retype[code]] is not None:
                    newVisit.append(label_ref[retype[code]])
            newVisit = list(set(newVisit))  # 消除列表中的重复元素
            newPatient.append(newVisit)
        label_seqs.append(newPatient)

    # pickle.dump(label_seqs, open(outFile + '.labelseqs', 'wb'), -1)
