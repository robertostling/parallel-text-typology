import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from mulres.embeddings import Embeddings


def print_dendogram(embedding_path, lang_set=None):
    language_embeddings = Embeddings(embedding_path)
    if lang_set is None:
        m = np.array([vec for vec in language_embeddings.embeddings.values()])
        labels = list(language_embeddings.embeddings.keys())
    else:
        m = np.array([vec for tr, vec in language_embeddings.embeddings.items() if tr in lang_set])
        labels = lang_set
    y = pdist(m, 'cosine')
    z = linkage(y, 'average')
    dn = dendrogram(z, labels=labels, leaf_font_size=6)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--embed_path', type=str, required=True)
    args = parser.parse_args()
    print_dendogram(args.embed_path)
