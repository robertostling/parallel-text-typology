import os
from multiprocessing import Pool

import mulres.config
from mulres.empfile import EncodedMPF
from mulres.utils import (load_resources_table, encoded_filename,
                          cached_embeddings_filename)
from mulres.embeddings import Embeddings


def cache_text(name, smith_filename):
    empf = EncodedMPF(encoded_filename(name))
    e = Embeddings(smith_filename, vocab=empf.sentences_vocab)
    print(name, 'has', len(e.embeddings), 'embeddings')
    out_filename = cached_embeddings_filename(name)
    e.write(out_filename)


def main():
    text_table = load_resources_table()

    os.makedirs(mulres.config.embeddings_cache_path, exist_ok=True)

    tasks = []
    for name, text in text_table.items():
        if 'smith' not in text: continue
        if text['preferred_source'] == 'no': continue
        filename = cached_embeddings_filename(name)
        if os.path.exists(filename): continue
        if not os.path.exists(encoded_filename(name)): continue
        smith_filename = mulres.config.smith_path_pattern % text['smith']
        if not os.path.exists(smith_filename): continue
        tasks.append((name, smith_filename))

    print(len(tasks), 'texts to cache embeddings for')

    with Pool() as p:
        p.starmap(cache_text, tasks, 1)

if __name__ == '__main__': main()

