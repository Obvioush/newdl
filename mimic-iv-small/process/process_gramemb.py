import numpy as np


def load_embedding(options):
    m = np.load(options)
    w = (m['w'] + m['w_tilde']) / 2.0
    return w


if __name__ == '__main__':
    # a = np.load('../resource/gram_emb/mimic4_gram.43.npz')
    #
    # knowledge_emb = a['W_emb'][8259:]

    # camp只需要顶点18个结点

    # np.save('../resource/gram_emb/gram_camp_knowledge_emb', knowledge_emb)

    glove_emb = load_embedding('../resource/gram_emb/mimic4_glove.14.npz')
    glove_knowledge_emb = glove_emb[6534:]
    # np.save('../resource/gram_emb/mimic4_glove_knowledge_emb', glove_knowledge_emb)




