import tensorflow as tf
import dgl
from dgl.data import CoraGraphDataset

dataset = CoraGraphDataset()
graph = dataset[0]
train_mask = graph.ndata['train_mask']
val_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']
labels = graph.ndata['label']
feat = graph.ndata['feat']