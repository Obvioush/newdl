import _pickle as pickle
import numpy as np
import tensorflow as tf

# treeFile = './resource/build_trees.level5.pk'
# treeMap = pickle.load(open(treeFile, 'rb'))
# ancestors = np.array(list(treeMap.values())).astype(np.int32)
# ancSize = ancestors.shape[1]
# leaves = []
# for k in treeMap.keys():
#     leaves.append([k] * ancSize)
# leaves = np.array(leaves).astype(np.int32)

tparams = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
leaves = np.array([0, 0, 0, 0])
ancestors = np.array([0, 1, 0, 2])
a = [tparams[leaves], tparams[ancestors]]
b = tf.keras.layers.concatenate(a, axis=1)
