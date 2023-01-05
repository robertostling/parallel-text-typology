"""Tools to read CoNLL-U files."""

import os.path
import gzip


def iterate_conllu(filename):
    """Read a CoNLL-U file.

    Returns:
        iterator over of (metadata, tokens), where metadata is a dict object
        with any "key: value" metadata occurring as comments before the
        sentence, and tokens is a list of lists, representing the fields of
        each token.  Compounds (multi-token items) are dropped.
    """
    tokens = []
    metadata = {}
    with (gzip.open(filename, 'rt', encoding='utf-8')
            if filename.endswith('.gz')
            else open(filename, 'r', encoding='utf-8')) as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    yield (metadata, tokens)
                    tokens = []
                    metadata = {}
            elif line.startswith('#'):
                fields = [s.strip() for s in line[1:].split(' = ', 1)]
                if len(fields) == 2:
                    metadata[fields[0]] = fields[1]
            else:
                fields = line.split('\t')
                # Drop multi-token rows
                if fields[0].isnumeric():
                    fields[0] = int(fields[0])
                    if fields[6] != '_':
                        fields[6] = int(fields[6])
                    tokens.append(fields)
        if tokens:
            raise ValueError('Expected blank line before EOF in '+filename)


def read_conllu(filename):
    return list(iterate_conllu(filename))

