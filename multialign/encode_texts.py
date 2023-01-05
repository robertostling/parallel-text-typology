import sys
import os.path
from multiprocessing import Pool

from mpfile import MPFile

def is_word(s):
    for c in s:
        if c.isalnum(): return True
    return False

def conllu_plain(fields):
    pos = fields[3]
    if pos in ('PUNCT', 'SYM'): return None
    return fields[1].casefold()

def conllu_pos_lemma(fields):
    pos = fields[3]
    if pos in ('PUNCT', 'SYM'): return None
    return pos + ':' + fields[2].casefold()

def is_nonsymbol(s):
    return s not in ('PUNCT', 'SYM')

def convert_file(args):
    filename, out_dir, verse_ids, pos_lemma = args
    name = os.path.splitext(os.path.basename(filename))[0]
    out_filename = os.path.join(out_dir, name + '.pickle.gz')
    conllu_transform = conllu_pos_lemma if pos_lemma else conllu_plain
    if os.path.exists(out_filename):
        print('File exists, skipping', name)
        return
    try:
        mpf = MPFile(filename, sent_ids=verse_ids, token_filter=is_word,
                     conllu_transform=conllu_transform)
        mpf.write_numpy(out_filename, verse_ids)
        print('Finished', mpf.name)
    except ValueError as e:
        print('Error processing', name)
        print(e)


def main():
    pos_lemma = dict(lemma=True, plain=False)[sys.argv[1]]
    verses_file = sys.argv[2]
    out_dir = sys.argv[3]
    filenames = sys.argv[4:]

    with open(verses_file) as f:
        verse_ids = list(map(str.strip, f))

    tasks = [(filename, out_dir, verse_ids, pos_lemma)
             for filename in filenames]
    print(len(tasks), 'texts to encode')
    with Pool() as p:
        p.map(convert_file, tasks, 1)


if __name__ == '__main__': main()

