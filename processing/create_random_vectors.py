import sys
import os

import numpy as np

import mulres.embeddings


def main():
    in_filename = sys.argv[1]
    out_filename = sys.argv[2]
    assert os.path.exists(in_filename)
    assert not os.path.exists(out_filename)

    e = mulres.embeddings.Embeddings(in_filename)
    #values = np.hstack([v for v in e.embeddings.values()])
    #mean = np.mean(values)
    #std = np.std(values)
    mean, std = 0.0, 1.0

    for k in e.embeddings:
        e.embeddings[k] = np.random.normal(mean, std, e.dim).astype(np.float32)

    e.write(out_filename)

if __name__ == '__main__': main()

