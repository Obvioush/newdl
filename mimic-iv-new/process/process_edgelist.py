import pickle
import pandas as pd
import dgl
import tensorflow as tf
import networkx as nx
# import matplotlib as plt
import scipy.sparse as sp


def tree_levelall():
    leaf2tree = pickle.load(open('../resource/mimic4.level5.pk', 'rb'))
    trees_l4 = pickle.load(open('../resource/mimic4.level4.pk', 'rb'))
    trees_l3 = pickle.load(open('../resource/mimic4.level3.pk', 'rb'))
    trees_l2 = pickle.load(open('../resource/mimic4.level2.pk', 'rb'))

    leaf2tree.update(trees_l4)
    leaf2tree.update(trees_l3)
    leaf2tree.update(trees_l2)

    return leaf2tree


# def visual(G):
#     # 可视化
#     nx_G = G.to_networkx().to_undirected()
#     pos = nx.kamada_kawai_layout(nx_G) ## 生成节点位置
#     nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
#     plt.pause(50)


if __name__ == '__main__':
    outFile = '../resource/mimic4'

    types = pickle.load(open('../resource/mimic4.types', 'rb'))
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

    temp = src_ids + dst_ids
    # temp.extend(dst_ids)
    code = list(set(temp))
    all = [ i for i in range(8987)]
    miss = list(set(all).difference(set(code)))
    misscode = []
    for i in miss:
        misscode.append(retype[i])

    srcc = []
    dstt = []
    for code in misscode:
        tokens = code.strip().split('.')
        prefix = ''
        for i in range(len(tokens)-1):
            if i == 0 or i == len(tokens)-1:
                prefix = prefix + tokens[i]
            else:
                prefix = prefix + '.' + tokens[i]
        srcc.append(int(types[prefix]))
        dstt.append(int(types[code]))

    src_ids.extend(srcc)
    dst_ids.extend(dstt)

    # g = dgl.graph((src_ids, dst_ids))
    # # 有向图转为无向图
    # bg = dgl.to_bidirected(g)
    # # print(g)
    #
    # # bbg = dgl.add_self_loop(bg)
    # # pickle.dump(bbg, open(outFile + '.graph', 'wb'), -1)
    # # nx.draw_networkx(g)
    # # visual(bg)
    #
    # nx_G = bg.to_networkx().to_undirected()
    # N = len(nx_G)
    # adj = nx.to_numpy_array(nx_G)
    # adj = sp.coo_matrix(adj)
    # pickle.dump(adj, open(outFile + '.adj', 'wb'), -1)


    # # 制作node2vec识别的edgelist
    # # 字典中的key值即为csv中列名
    # edgedata = pd.DataFrame({'src': src_ids, 'dst': dst_ids})
    # # edgedata_re = pd.DataFrame({'src': dst_ids, 'dst': src_ids})
    #
    # # # 将DataFrame存储为csv,index表示是否显示行名，default=True
    # # dataframe.to_csv("edgelist.csv", index=False, sep=',')
    # #
    # # # csv转txt
    # # data = pd.read_csv('news_data.csv', encoding='utf-8')
    # with open('../resource/mimic4_edgelist.txt', 'a+', encoding='utf-8') as f:
    #     for line in edgedata.values:
    #         f.write((str(line[0]) + '\t' + str(line[1]) + '\n'))
    #     # for line in edgedata_re.values:
    #     #     f.write((str(line[0]) + '\t' + str(line[1]) + '\n'))
