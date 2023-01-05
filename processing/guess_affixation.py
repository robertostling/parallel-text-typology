"""Script to guess affixation type given paradigms"""

import sys
import os.path
from multiprocessing import Pool
import gzip
from collections import defaultdict, Counter
from operator import itemgetter
import statistics
import json

import Levenshtein

import mulres.utils


def count_transforms(forms, counts):
    for i, form1 in enumerate(forms):
        for form2 in forms[i+1:]:
            ops = Levenshtein.editops(form1, form2)
            seg1 = [0, 0, '']
            seg2 = [0, 0, '']
            segs1, segs2 = [], []
            for op, p1, p2 in ops:
                if op in ('insert', 'replace'):
                    # character inserted at form2[p2] (or replaced)
                    if p2 > seg2[1]+1:
                        segs2.append(seg2)
                        seg2 = [p2, p2, form2[p2]]
                    else:
                        seg2[1] = p2
                        seg2[2] += form2[p2]
                if op in ('delete', 'replace'):
                    # character deleted at form1[p1] (or replaced)
                    if p1 > seg1[1]+1:
                        segs1.append(seg1)
                        seg1 = [p1, p1, form1[p1]]
                    else:
                        seg1[1] = p1
                        seg1[2] += form1[p1]

            segs1.append(seg1)
            segs2.append(seg2)

            for segs, n in ((segs1, len(form1)), (segs2, len(form2))):
                for start, end, s in segs:
                    if not s: continue
                    half = 0 if start < n/2 else 1
                    counts[(s, half)] += 1


def guess_affixation(text):
    filename = mulres.utils.paradigms_filename(text['name'])
    pos_transforms = defaultdict(Counter)
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            fields = line.split()
            pos = fields[0]
            forms = fields[1:]
            count_transforms(forms, pos_transforms[pos])

    pos_data = {}

    for pos, transforms in pos_transforms.items():
        table = sorted(transforms.items(), key=itemgetter(1), reverse=True)

        mean_top = statistics.mean([n for _, n in table[:3]])

        if mean_top < 30:
            # Probably no significant affixation, empirically determined
            # threshold
            pos_data[pos] = dict(prefixes=[], suffixes=[])
        else:
            # Counts for prefixes, suffixes
            counts = [[], []]
            for (s, half), n in table:
                if n < mean_top / 4: break
                counts[half].append((s, n))

            pos_data[pos] = dict(prefixes=counts[0],
                                 suffixes=counts[1])

    return pos_data


def main():
    text_info = mulres.utils.load_resources_table()

    texts = [text for name, text in text_info.items()
             if os.path.exists(mulres.utils.paradigms_filename(name))]

    texts.sort(key=lambda text: text['name'])

    data = {}
    for text in texts:
        data[text['name']] = guess_affixation(text)
        print(text['name'])

    with open(mulres.config.affixation_path, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)


if __name__ == '__main__': main()

