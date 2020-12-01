import pickle
from functools import reduce


def process_newTrees(dataseqs, tree_old):
    leaf2tree = pickle.load(open('../resource/mimic4.level5.pk', 'rb'))
    trees_l4 = pickle.load(open('../resource/mimic4.level4.pk', 'rb'))
    trees_l3 = pickle.load(open('../resource/mimic4.level3.pk', 'rb'))
    trees_l2 = pickle.load(open('../resource/mimic4.level2.pk', 'rb'))
    tree_seq = []

    leaf2tree.update(trees_l4)
    leaf2tree.update(trees_l3)
    leaf2tree.update(trees_l2)

    for patient in dataseqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in leaf2tree[code]:
                    leaf2tree[code].remove(code)
                    newVisit.append(leaf2tree[code])
                else:  # 表明code已经在上一个if中被删掉
                    newVisit.append(leaf2tree[code])
            newVisit = list(set(reduce(lambda x,y:x+y, newVisit)))  # reduce将多个数组合并
            newPatient.append(newVisit)
        tree_seq.append(newPatient)

    newTreeseq = []
    for patient in tree_seq:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                newVisit.append(tree_old[code])
            newPatient.append(newVisit)
        newTreeseq.append(newPatient)

    return newTreeseq


def process_Trees(dataseqs):
    leaf2tree = pickle.load(open('../resource/mimic4.level5.pk', 'rb'))
    trees_l4 = pickle.load(open('../resource/mimic4.level4.pk', 'rb'))
    trees_l3 = pickle.load(open('../resource/mimic4.level3.pk', 'rb'))
    trees_l2 = pickle.load(open('../resource/mimic4.level2.pk', 'rb'))
    tree_seq = []

    leaf2tree.update(trees_l4)
    leaf2tree.update(trees_l3)
    leaf2tree.update(trees_l2)

    for patient in dataseqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in leaf2tree[code]:
                    leaf2tree[code].remove(code)
                    newVisit.append(leaf2tree[code])
                else:
                    newVisit.append(leaf2tree[code])
            newVisit = list(set(reduce(lambda x,y:x+y, newVisit)))
            newPatient.append(newVisit)
        tree_seq.append(newPatient)
    return tree_seq


if __name__ == '__main__':
    outFile1 = '../resource/mimic4_tree'
    outFile2 = '../resource/mimic4_newTree'

    data_seqs = pickle.load(open('../resource/mimic4.seqs', 'rb'))
    trees_type = pickle.load(open('../resource/mimic4.types', 'rb'))
    retype = dict([(v, k) for k, v in trees_type.items()])

    treenode = {}  # 映射祖先节点新数组和节点ccs_multi的信息
    tree2old = {}  # 映射祖先新数组与旧数组的index对应关系
    count = 0

    # 共有728个祖先节点(包含root节点和ccs分类节点)
    for i in range(8259, len(retype)):
        treenode[count] = retype[i]
        tree2old[i] = count
        count += 1

    newTreeSeq = process_newTrees(data_seqs, tree2old)  # 下标范围从0到727, 共728个节点
    # treeSeq = process_Trees(data_seqs)  # 下标范围从8259到8986, 共728个节点

    # pickle.dump(treeSeq, open(outFile1 + '.seqs', 'wb'), -1)
    # pickle.dump(newTreeSeq, open(outFile2 + '.seqs', 'wb'), -1)
