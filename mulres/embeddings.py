"""Class for reading GloVe-style embeddings"""

from operator import itemgetter
import numpy as np

class Embeddings:
    def __init__(self, filename, limit=None, vocab=None):
        embeddings = {}
        with open(filename, 'r', encoding='utf-8') as f:
            line = next(f)
            fields = line.split()
            if len(fields) == 2:
                total, dim = int(fields[0]), int(fields[1])
            else:
                word, vec = line.split(' ', 1)
                vec = np.fromstring(vec, sep=' ', dtype=np.float32)
                dim = len(vec)
                if not (vocab and word not in vocab):
                    embeddings[word] = vec

            for line in f:
                if limit and len(embeddings) >= limit: break
                word, vec = line.split(' ', 1)
                if vocab and word not in vocab: continue
                if word == '<pad_lang>': continue
                vec = np.fromstring(vec, sep=' ', dtype=np.float32)
                assert dim == len(vec), \
                        ('expected %d dimensions, got %d for %s' % (
                            dim, len(vec), word))
                embeddings[word] = vec
        self.embeddings = embeddings
        self.dim = dim

    def __getitem__(self, x):
        #if x not in self.embeddings:
        #    return np.zeros((self.dim,), dtype=np.float32)
        return self.embeddings.get(x)

    def __contains__(self, x):
        return x in self.embeddings

    def write(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            print(len(self.embeddings), self.dim, file=f)
            for word, vec in sorted(self.embeddings.items(), key=itemgetter(0)):
                if any(c.isspace() for c in word): continue
                print(' '.join([word] + ['%.5f' % x for x in vec]), file=f)

