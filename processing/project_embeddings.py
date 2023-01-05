import sys
import gzip
import pickle
from multiprocessing import Pool
from collections import Counter, defaultdict
import os
import random

import numpy as np
from scipy.spatial.distance import pdist, cdist

import mulres.config
from mulres.utils import (load_resources_table, encoded_filename,
                          aligned_filename, embeddings_filename,
                          cached_embeddings_filename)
from mulres.empfile import EncodedMPF
from mulres.embeddings import Embeddings

# TODO: perhaps shouldn't be hardcoded...
EMBEDDINGS_DIM = 300

def token_alignments(source_empf, target_empf, lemma_candidates):
    assert source_empf.verse_ids == target_empf.verse_ids

    token_translations = defaultdict(Counter)

    for verse_i, verse in enumerate(source_empf.verse_ids):
        if source_empf.sentences[verse_i] is None: continue
        if target_empf.sentences[verse_i] is None: continue

        # target n-gram mapped to list of indexes in target verse
        ngram_positions = defaultdict(list)

        target_tokens = [target_empf.sentences_vocab[token_i]
                         for token_i in target_empf.sentences[verse_i]]
        source_tokens = [source_empf.sentences_vocab[token_i]
                         for token_i in source_empf.sentences[verse_i]]
        source_lemmas = [source_empf.lemma_vocab[token_i]
                         for token_i in source_empf.lemma[verse_i]]

        assert len(source_tokens) == len(source_lemmas)

        target_links = [None]*len(target_tokens)

        for token_idx, token in enumerate(target_tokens):
            # token plus boundary markers
            s = '#' + token + '#'
            for i in range(len(s)-1):
                for j in range(i+2, len(s)+1):
                    ngram = s[i:j]
                    ngram_positions[ngram].append(token_idx)

        for lemma_i, (lemma, plain) in enumerate(
                zip(source_lemmas, source_tokens)):
            candidates = lemma_candidates[lemma]
            # candidates should be sorted by decreasing saved_per_token
            for form, bayes_factor, saved_per_token in candidates:
                # TODO: adjust thresholds?
                if bayes_factor < 0.0:
                    continue
                elif saved_per_token < 0.7 and bayes_factor < 100:
                    continue
                elif saved_per_token < 0.2:
                    break
                # the scores of this form seems acceptable, find the first
                # unaligned match for it
                for idx in ngram_positions[form]:
                    link = target_links[idx]
                    if link is None or saved_per_token > link[1]:
                        target_links[idx] = (plain, saved_per_token)
                        break

        for target, link in zip(target_tokens, target_links):
            if link is not None:
                plain, _ = link
                token_translations[target][plain] += 1

    return dict(token_translations)

def project_embeddings(target_text, text_embeddings):
    e_filename = embeddings_filename(target_text['name'])
    if os.path.exists(e_filename):
        print(e_filename, 'exists, skipping')
        return

    target_empf = EncodedMPF(encoded_filename(target_text['name']))
    with gzip.open(aligned_filename(target_text['name']), 'rb') as f:
        source_names = pickle.load(f)
        candidates_list = pickle.load(f)

    target_vectors = defaultdict(lambda: np.zeros(EMBEDDINGS_DIM, np.float))
    target_n_aligned = Counter()

    #target_all_vectors = defaultdict(list)

    for source_name, lemma_candidates in zip(source_names, candidates_list):
        if source_name not in text_embeddings:
            #print('Skipping, since no embeddings for', source_name)
            continue
        source_empf = EncodedMPF(encoded_filename(source_name))
        alignments = token_alignments(
                source_empf, target_empf, lemma_candidates)

        #print(source_name,
        #        'yields', len(alignments),
        #        'word forms aligned, of', len(target_empf.sentences_vocab))

        source_vocabulary = sorted(
                {word for counts in alignments.values()
                      for word in counts.keys()})
        source_e = Embeddings(text_embeddings[source_name],
                              vocab=source_vocabulary)

        #print('Loaded', len(source_e.embeddings), 'embeddings for',
        #        source_name[:3], 'of', len(source_vocabulary), 'requested')

        #missing = set(source_e.embeddings.keys()) - set(source_vocabulary)
        #print('Missing words include:', ' '.join(random.sample(missing, 10)))

        for target_token, source_counts in alignments.items():
            for source_token, n in source_counts.items():
                v = source_e[source_token]
                if v is not None:
                    target_n_aligned[target_token] += n
                    target_vectors[target_token] += v*n

            #for source_token, n in source_counts.items():
            #    v = source_e[source_token]
            #    if v is not None:
            #        target_all_vectors[target_token].append((v, n))
            #        break

    e = {token: v/target_n_aligned[token]
         for token, v in target_vectors.items()}

    #all_d = []
    #for token, all_vectors in target_all_vectors.items():
    #    if len(all_vectors) < 2: continue
    #    v = np.array([v for v, _ in all_vectors])
    #    d = pdist(v, 'cosine')
    #    all_d.append(d)
    #    #print(token, round(np.mean(d), 3), '+/-', round(np.std(d), 3))

    #mixed_d = []
    #all_tokens = list(target_all_vectors.keys())
    #for token1, all_vectors in target_all_vectors.items():
    #    if len(all_vectors) < 2: continue
    #    token2 = random.choice(all_tokens)
    #    all_other_vectors = target_all_vectors[token2]
    #    if len(all_other_vectors) < 2: continue
    #    v = np.array([v for v, _ in all_vectors])
    #    u = np.array([v for v, _ in all_other_vectors])
    #    d = cdist(u, v, 'cosine').flatten()
    #    mixed_d.append(d)

    #d = np.concatenate(all_d)
    #print('ACTUAL', target_text['name'], round(np.mean(d), 3), '+/-', round(np.std(d), 3))
    #d = np.concatenate(mixed_d)
    #print('BASELINE', target_text['name'], round(np.mean(d), 3), '+/-', round(np.std(d), 3))

    print('Writing embeddings for', target_text['name'])
    with open(e_filename, 'w', encoding='utf-8') as f:
        print(len(target_vectors), EMBEDDINGS_DIM, file=f)
        for token in sorted(target_vectors.keys()):
            if any(c.isspace() for c in token): continue
            v = target_vectors[token]/target_n_aligned[token]
            print(' '.join([token] + ['%.5f' % x for x in v]), file=f)


def main():
    text_table = load_resources_table()

    os.makedirs(mulres.config.embeddings_path, exist_ok=True)

    target_texts = [
            info for name, info in text_table.items()
            if os.path.exists(encoded_filename(name)) and
               os.path.exists(aligned_filename(name))]

    text_embeddings = {
        name: cached_embeddings_filename(name)
        for name, text in text_table.items()
        if os.path.exists(cached_embeddings_filename(name))}

    print('Found embeddings for', len(text_embeddings), 'source texts')
    print(len(target_texts), 'texts to project embeddings for')

    tasks = [(text, text_embeddings) for text in target_texts]
    with Pool() as p:
        p.starmap(project_embeddings, tasks)


if __name__ == '__main__': main()

