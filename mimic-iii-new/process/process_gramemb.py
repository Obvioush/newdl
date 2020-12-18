import numpy as np


def load_embedding(options):
    m = np.load(options)
    w = (m['w'] + m['w_tilde']) / 2.0
    return w


if __name__ == '__main__':
    a = np.load('../resource/gram_emb/mimic3_gram.33.npz')

    knowledge_emb = a['W_emb'][4880:]

    # camp只需要顶点18个结点

    # np.save('../resource/gram_emb/gram_camp_knowledge_emb', knowledge_emb)

    glove_emb = load_embedding('../resource/gram_emb/mimic3_glove.22.npz')
    glove_knowledge_emb = glove_emb[4880:]
    np.save('../resource/gram_emb/mimic3_glove_knowledge_emb', glove_knowledge_emb)




