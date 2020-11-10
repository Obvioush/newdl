import numpy as np
import _pickle as pickle

seqFile = './resource/mimic3.seqs'
typeFile = './resource/mimic3.types'
seqs = pickle.load(open(seqFile, 'rb'))
types = pickle.load(open(typeFile, 'rb'))



