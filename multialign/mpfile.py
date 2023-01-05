import re
import os.path
import gzip
from collections import OrderedDict
import pickle

import numpy as np

import re

from conllu import iterate_conllu

RE_WORD = re.compile('\w')

class MPFile:
    """A file from the paralleltext repository

    Metadata is stored in the 'metadata' attribute as a dict, while the actual
    parallel sentences are stored in 'sentences' as an identifier: sentence 
    mapping. Note that metadata keys are made lower-case.
    """

    def __init__(self, filename=None, **kwargs):
        self.metadata = OrderedDict()
        self.sentences = OrderedDict()
        if filename is not None:
            if filename.endswith('.conllu') or filename.endswith('.conllu.gz'):
                self.read_conllu(filename, **kwargs)
            else:
                self.read(filename, **kwargs)

    def write_numpy(self, path, sent_ids):
        token_list = sorted({
            token.casefold() for tokens in self.sentences.values()
                             for token in tokens})
        token_index = {token: i for i, token in enumerate(token_list)}

        verse_tokens = []
        for sent_id in sent_ids:
            if sent_id not in self.sentences:
                tokens = None
            else:
                tokens = np.array([token_index[token.casefold()]
                                   for token in self.sentences[sent_id]],
                                  dtype=np.int32)
            verse_tokens.append(tokens)

        with gzip.open(path, 'wb') as f:
            pickle.dump(token_list, f)
            pickle.dump(verse_tokens, f)


    def write(self, path, file_format, write_metadata=True):
        """Write the contents of this object to a file"""
        filename = os.path.join(path, '.'.join([self.name, file_format]))
        if file_format == 'txt.gz':
            with gzip.open(filename, 'wt', encoding='utf-8') as f:
                if write_metadata:
                    for k, v in self.metadata.items():
                        print('# {}: {}'.format(k, v), file=f)
                for k, v in self.sentences.items():
                    print(k + '\t' + ' '.join(v), file=f)
        elif file_format == 'turku':
            with open(filename, 'w', encoding='utf-8') as f:
                for k, v in self.sentences.items():
                    print('###C: verse = {}'.format(k), file=f)
                    print(' '.join(v), file=f)
        else: raise NotImplemented('Unknown file format: ' + file_format)

    def write_bitext(self, other, filename, index_filename, punctuation=False):
        common = sorted(set(self.sentences.keys()) &
                        set(other.sentences.keys()))
        with open(filename, 'w', encoding='utf-8') as outf, \
             open(index_filename, 'w') as indexf:
            for k in common:
                v1 = self.sentences[k]
                v2 = other.sentences[k]
                if not punctuation:
                    v1 = [token for token in v1 if RE_WORD.search(token)]
                    v2 = [token for token in v2 if RE_WORD.search(token)]
                if v1 and v2:
                    assert all(' ' not in token for token in v1)
                    assert all(' ' not in token for token in v2)
                    print(k, file=indexf)
                    print(' '.join(v1) + ' ||| ' + ' '.join(v2), file=outf)


    def read_conllu(self, filename, sent_ids=None, token_filter=None,
                    conllu_transform=(lambda fields: fields[1])):
        """Fill this object with data from a CoNLL-U file

        This could be either one of the manually annotated PROIEL-derived
        files from the paralleltext repo, or a Turku Neural Parsing Pipeline
        parsed file with 'verse = xxx' metadata.

        Only lemmas are kept, this is mainly for use by encode_texts.py
        """
        verse = None
        self.name = os.path.basename(filename).split('.', 1)[0]
        for metadata, tokens in iterate_conllu(filename):
            verse = metadata.get('verse', verse)
            for fields in tokens:
                misc = dict(kv.split('=') for kv in fields[9].split('|')
                            if '=' in kv)
                verse = misc.get('ref', verse)
                if verse is None:
                    raise ValueError('CoNLL-U file without verse identifiers')
                if sent_ids is None or verse in sent_ids:
                    result = conllu_transform(fields)
                    if (token_filter is None or token_filter(fields[1])) and \
                       result is not None:
                        pos = fields[3]
                        token = pos+':'+(fields[2].casefold())
                        if verse not in self.sentences:
                            self.sentences[verse] = []
                        self.sentences[verse].append(result)

    def read(self, filename, only_metadata=False, sent_ids=None,
            token_filter=None, conllu_transform=None):
        """Fill this object with data from a paralleltext format file.

        This modifies the `metadata` and `sentences` attributes.
        """
        if sent_ids is not None and not isinstance(sent_ids, set):
            sent_ids = set(sent_ids)
        self.name = os.path.basename(filename).split('.', 1)[0]
        re_metadata = re.compile(r'#\s*([^:]+):\s*(.*)$')
        with gzip.open(filename, 'rt', encoding='utf-8') \
                if filename.endswith('.gz') else \
                open(filename, 'r', encoding='utf-8') as f:
            for i,line in enumerate(f):
                line = line.strip()
                if line.startswith('#'):
                    m = re_metadata.match(line)
                    if m:
                        self.metadata[m.group(1).lower()] = m.group(2)
                else:
                    if only_metadata: return
                    fields = line.split('\t')
                    if len(fields) == 2:
                        sent_id, sent = fields
                        if sent_ids is None or sent_id in sent_ids:
                            tokens = sent.split()
                            if token_filter is None:
                                self.sentences[sent_id] = tokens
                            else:
                                self.sentences[sent_id] = [
                                        token for token in tokens
                                        if token_filter(token)]
                    elif len(fields) < 2:
                        pass
                    else:
                        raise ValueError(
                        'Expected comment or two-column line at %s:%d' % (
                            filename, i+1))
        if (not self.sentences) and (not only_metadata):
            raise ValueError('No valid sentences in file: ' + filename)

