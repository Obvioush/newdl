import pickle
import dgl
import tensorflow as tf


def tree_levelall():
    leaf2tree = pickle.load(open('../resource/mimic3.level5.pk', 'rb'))
    trees_l4 = pickle.load(open('../resource/mimic3.level4.pk', 'rb'))
    trees_l3 = pickle.load(open('../resource/mimic3.level3.pk', 'rb'))
    trees_l2 = pickle.load(open('../resource/mimic3.level2.pk', 'rb'))

    leaf2tree.update(trees_l4)
    leaf2tree.update(trees_l3)
    leaf2tree.update(trees_l2)

    return leaf2tree


if __name__ == '__main__':
    # outFile1 = '../resource/mimic3_tree'

    types = pickle.load(open('../resource/mimic3.types', 'rb'))
    retype = dict([(v, k) for k, v in types.items()])

    edgelist = {}
    tree = tree_levelall()

    for key, value in tree.items():
        for index, node in enumerate(value):
            if index == 0:
                continue
            if index == len(value) - 1:
                if node not in edgelist:
                    edgelist[node] = [key]
                else:
                    edgelist[node].append(key)
            else:
                if node not in edgelist:
                    edgelist[node] = [value[index+1]]
                elif value[index+1] not in edgelist[node]:
                    edgelist[node].append(value[index+1])

    # Source nodes for edges
    src_ids = []
    # Destination nodes for edges
    dst_ids = []

    for key, value in edgelist.items():
        src_ids.extend([key for i in range(len(value))])
        dst_ids.extend(value)

    g = dgl.graph((src_ids, dst_ids))
    # 有向图转为无向图
    bg = dgl.to_bidirected(g)
    print(g)
