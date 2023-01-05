import sys
import os
from multiprocessing import Pool
import traceback

from mulres.mpfile import MPFile, is_word
from mulres.utils import load_resources_table, load_verse_table
import mulres.config

def convert_file(info, filename, out_dir, verse_ids):
    name = info['name']
    out_filename = os.path.join(out_dir, name + '.gz')
    if os.path.exists(out_filename):
        print('File exists, skipping', name)
        return
    try:
        mpf = MPFile(filename, sent_ids=verse_ids, token_filter=is_word)
        mpf.write_numpy(out_filename, verse_ids)
        print('Finished', mpf.name)
    except ValueError as e:
        print('Error processing', name)
        print(e)
        print(traceback.format_exc())
        if os.path.exists(out_filename):
            os.remove(out_filename)


def main():
    text_table = load_resources_table()
    verse_ids = load_verse_table()

    os.makedirs(mulres.config.encoded_path, exist_ok=True)

    tasks = []
    for name, info in text_table.items():
        # If there is no complete New Testament, don't even encode text
        if info['nt_verses'] < len(verse_ids)*0.8: continue
        has_ud = (info['preferred_source'] == 'yes' and 'ud' in info)
        if has_ud:
            filename = os.path.join(mulres.config.ud_path, name+'.conllu')
        else:
            filename = os.path.join(mulres.config.corpus_path, name+'.txt.gz')
        if os.path.exists(filename):
            tasks.append(
                    (info, filename, mulres.config.encoded_path, verse_ids))

    print(len(tasks), 'texts to encode')
    with Pool() as p:
        p.starmap(convert_file, tasks, 1)


if __name__ == '__main__': main()

