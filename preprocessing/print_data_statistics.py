import sys
from collections import defaultdict, Counter
import gzip
import pickle
from pprint import pprint
import os.path

import mulres.config
import mulres.utils
from mulres.ids import IDS


def main():
    n_sources = 0
    n_total_texts = 0
    n_ids = 0
    n_aligned = 0
    ids_iso_count = Counter()
    ids_preferred = set()
    all_sources = set()
    aligned_isos = set()
    all_isos = set()
    encoded_isos = set()
    n_encoded = 0

    ids = IDS(mulres.config.ids_path)

    table = mulres.utils.load_resources_table()
    for name, info in table.items():
        iso = info['iso']
        all_isos.add(iso)
        if os.path.exists(mulres.utils.encoded_filename(name)):
            n_encoded += 1
            encoded_isos.add(iso)
        if iso in ids.isos:
            ids_iso_count[iso] += 1
        if info.get('preferred_source') == 'yes':
            n_sources += 1
            if iso in ids.isos: ids_preferred.add(name)
        n_total_texts += 1

        aligned_filename = mulres.utils.aligned_filename(info['name'])
        if os.path.exists(aligned_filename):
            n_aligned += 1
            aligned_isos.add(iso)
            with gzip.open(aligned_filename, 'rb') as f:
                source_names = pickle.load(f)
            all_sources |= set(source_names)


    pprint(sorted(all_sources))

    print(f'{n_sources} preferrred sources')
    print(f'{n_total_texts} texts in total in {len(all_isos)} languages')
    print(f'{n_encoded} selected texts in total in {len(encoded_isos)} languages')
    print(f'{len(ids_iso_count)} languages ({sum(ids_iso_count.values())} texts) with IDS lexicon')
    print(f'{len(ids_preferred)} IDS sources')
    print(f'{n_aligned} aligned target texts in {len(aligned_isos)} languages')
    print(f'{len(all_sources)} aligned source texts')

if __name__ == '__main__': main()


