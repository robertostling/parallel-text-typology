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
import mulres.utils
from mulres.empfile import EncodedMPF
from mulres.embeddings import Embeddings
from mulres.ids import IDS


def project_one(source_empf, target_empf, lemma_candidates,
                lemma_concepts,
                verse_proj_pos, verse_proj_dep, verse_proj_ids):
    assert source_empf.verse_ids == target_empf.verse_ids

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
        source_pos = [source_empf.pos_vocab[token_i]
                      for token_i in source_empf.pos[verse_i]]
        source_head = source_empf.head[verse_i]
        source_dep = [source_empf.dep_vocab[token_i]
                      for token_i in source_empf.dep[verse_i]]

        assert len(source_tokens) == len(source_lemmas)

        target_links = [None]*len(target_tokens)

        for token_idx, token in enumerate(target_tokens):
            # token plus boundary markers
            s = '#' + token + '#'
            for i in range(len(s)-1):
                for j in range(i+2, len(s)+1):
                    ngram = s[i:j]
                    ngram_positions[ngram].append(token_idx)

        lemma_count = Counter(source_lemmas)

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
                        target_links[idx] = (plain, saved_per_token, lemma_i)
                        break

        source_links = [None]*len(source_tokens)
        for target_i, link in enumerate(target_links):
            if link is not None:
                _, _, source_i = link
                source_links[source_i] = target_i

        for source_i, target_i in enumerate(source_links):
            if target_i is None: continue
            lemma = source_lemmas[source_i]
            # TODO: also check lemma of head?
            if lemma_count[lemma] > 1: continue
            pos = source_pos[source_i]
            dep = source_dep[source_i].split(':')[0]
            head = source_head[source_i] if source_head[source_i] < 0 \
                    else source_links[source_head[source_i]]

            if lemma_concepts is not None:
                for ids in lemma_concepts.get(lemma, ()):
                    verse_proj_ids[verse_i][target_i][ids] += 1
            verse_proj_pos[verse_i][target_i][pos] += 1
            if head is not None:
                verse_proj_dep[verse_i][(target_i, head, dep)] += 1


def project_wordorder(target_text):
    target_empf = EncodedMPF(
            mulres.utils.encoded_filename(target_text['name']))
    aligned_filename = mulres.utils.aligned_filename(target_text['name'])
    with gzip.open(aligned_filename, 'rb') as f:
        source_names = pickle.load(f)
        candidates_list = pickle.load(f)

    ids_pos_table = mulres.utils.load_ids_pos_table()

    def core_pos(pos, ids):
        # pos is a UD POS tag, or None
        # ids is an IDS concept ID (str), or None
        if pos is None or ids is None: return pos
        if ids in ids_pos_table: return ids_pos_table[ids] + ':CORE'
        return pos

    ids = IDS(mulres.config.ids_path)

    verse_proj_pos = defaultdict(lambda: defaultdict(Counter))
    verse_proj_dep = defaultdict(Counter)
    verse_proj_ids = defaultdict(lambda: defaultdict(Counter))

    # map from (pos, head_pos, label) to [head_initial, head_final] counts
    # word_order uses UD's POS tags
    # core_word_order uses prototypical POS tags if available, with UD as
    # fallback
    word_order = defaultdict(lambda: [0, 0])
    core_word_order = defaultdict(lambda: [0, 0])

    # XXX --------------------------------------------------------------
    #source_names = source_names[:6]
    #candidates_list = candidates_list[:6]
    # XXX --------------------------------------------------------------

    n_ids_sources = 0
    for source_name, lemma_candidates in zip(source_names, candidates_list):
        source_empf = EncodedMPF(mulres.utils.encoded_filename(source_name))
        source_iso = source_name[:3]
        n_ids_sources += int(source_iso in ids.isos)
        project_one(
                source_empf, target_empf, lemma_candidates,
                ids[source_iso] if source_iso in ids.isos else None,
                verse_proj_pos, verse_proj_dep, verse_proj_ids)

    n_sources = len(source_names)
    # Minimum number of sources that need to agree on an analysis
    min_pos_sources = n_sources / 5
    min_dep_sources = n_sources / 5
    min_ids_sources = n_ids_sources / 5
    for verse_i in range(target_empf.n_verses):
        if target_empf.sentences[verse_i] is None: continue
        target_tokens = [target_empf.sentences_vocab[token_i]
                         for token_i in target_empf.sentences[verse_i]]
        target_pos = [None]*len(target_tokens)

        proj_pos = verse_proj_pos.get(verse_i)
        proj_dep = verse_proj_dep.get(verse_i)
        proj_ids = verse_proj_ids.get(verse_i)
        if proj_pos is None or proj_dep is None: continue

        target_head_opts = [[] for _ in target_tokens]
        for (dep_i, head_i, label), n in proj_dep.items():
            target_head_opts[dep_i].append((n, head_i, label))

        target_pos = [None]*len(target_tokens)
        target_head = [None]*len(target_tokens)
        target_label = [None]*len(target_tokens)
        target_ids = [None]*len(target_tokens)
        #print(target_empf.verse_ids[verse_i])
        for token_i in range(len(target_tokens)):
            pos_counts = proj_pos.get(token_i)
            ids_counts = None if proj_ids is None else proj_ids.get(token_i)
            pos = None
            label = None
            head = None
            ids = None
            if pos_counts is not None:
                pos, n = pos_counts.most_common(1)[0]
                if n < min_pos_sources: pos = None
            if ids_counts is not None:
                ids, n = ids_counts.most_common(1)[0]
                if n < min_ids_sources: ids = None
            target_head_opts[token_i].sort(reverse=True)
            if target_head_opts[token_i] and \
                    target_head_opts[token_i][0][0] >= min_dep_sources:
                n, head, label = target_head_opts[token_i][0]
            target_pos[token_i] = pos
            target_head[token_i] = head
            target_label[token_i] = label
            target_ids[token_i] = ids

        for token_i, (token, pos, head, label, ids) in enumerate(zip(
            target_tokens, target_pos, target_head, target_label, target_ids)):
            #print(token_i, token, pos, head, label, ids)

            if (pos is not None) and (label is not None) and \
                    head >= 0 and target_pos[head] is not None:
                head_final = 1 if token_i < head else 0
                head_pos = target_pos[head]
                head_ids = target_ids[head]
                word_order[(pos, head_pos, label)][head_final] += 1
                core_tuple = (
                        core_pos(pos, ids),
                        core_pos(head_pos,  head_ids),
                        label)
                core_word_order[core_tuple][head_final] += 1

    word_order = sorted(word_order.items(), key=lambda t: t[0])
    core_word_order = sorted(core_word_order.items(), key=lambda t: t[0])

    filename = mulres.utils.wordorder_filename(target_text['name'])
    with open(filename, 'w', encoding='utf-8') as f:
        for (pos, head_pos, label), (hi, hf) in core_word_order:
            print('\t'.join([pos, head_pos, label, str(hi), str(hf)]),
                    file=f)
        #for (pos, head_pos, label), (hi, hf) in word_order:
        #    print('\t'.join(['ud', pos, head_pos, label, str(hi), str(hf)]),
        #            file=f)
        #    print('%-8s %-8s %-12s %3d %3d' % (pos, head_pos, label, hi, hf))

def main():
    text_table = mulres.utils.load_resources_table()

    os.makedirs(mulres.config.wordorder_path, exist_ok=True)

    target_texts = [
            info for name, info in text_table.items()
            if os.path.exists(mulres.utils.encoded_filename(name)) and
               os.path.exists(mulres.utils.aligned_filename(name))]

    #target_texts = [info for info in target_texts
    #                if info['name'] in ('swe-x-bible-2000', 'fao-x-bible')]

    print(len(target_texts), 'texts to project word order statistics for')

    with Pool() as p:
        p.map(project_wordorder, target_texts)


if __name__ == '__main__': main()

