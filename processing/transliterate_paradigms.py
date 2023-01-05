from multiprocessing import Pool
import os
import gzip

import mulres.transliterate
import mulres.utils
from mulres.empfile import EncodedMPF

def transliterate_text(name):
    empf = EncodedMPF(mulres.utils.encoded_filename(name))
    filename = mulres.utils.transliterated_filename(name)
    with gzip.open(filename, 'wt', encoding='utf-8') as f:
        for sentence in empf.sentences:
            if sentence is None: continue
            text = ' '.join(empf.sentences_vocab[i] for i in sentence)
            transliterated = mulres.transliterate.remove_distinctions(
                    mulres.transliterate.normalize(text))
            print(transliterated, file=f)
    print('Transliterated', name, flush=True)


def main():
    text_info = mulres.utils.load_resources_table()
    tasks = []
    for name, info in text_info.items():
        filename = mulres.utils.encoded_filename(name)
        if os.path.exists(filename):
            tasks.append(name)

    os.makedirs(mulres.config.transliterated_path, exist_ok=True)

    with Pool() as p:
        p.map(transliterate_text, tasks, 1)


if __name__ == '__main__': main()

