import argparse
import multiprocessing
import os
import random
import time
from multiprocessing import Pool

from tqdm import tqdm

from mulres import config
from mulres import transliterate
from mulres.mpfile_light import MPFile


def read_and_transliterate(bible_path):
    bib = MPFile(bible_path + ".txt.gz")
    bib.sentences = {id: " ".join(
        list(transliterate.remove_distinctions(transliterate.normalize(sent)).replace(" ", "$"))) for id, sent in bib.sentences.items()}
    return {bible_path.split("/")[-1]: bib}


def create_nmt_data(bibles, out_dir, num_of_batch, target_bible, bs=100):
    s = time.time()
    print("Training data preparation")

    out_dir_to_eng = os.path.join(out_dir, "others_to_eng")
    out_dir_from_eng = os.path.join(out_dir, "eng_to_others")

    os.makedirs(out_dir_to_eng)
    os.makedirs(out_dir_from_eng)

    to_eng_src_train_dir = os.path.join(out_dir_to_eng, "src-train.txt")
    to_eng_tgt_train_dir = os.path.join(out_dir_to_eng, "tgt-train.txt")
    to_eng_src_val_dir = os.path.join(out_dir_to_eng, "src-val.txt")
    to_eng_tgt_val_dir = os.path.join(out_dir_to_eng, "tgt-val.txt")

    from_eng_src_train_dir = os.path.join(out_dir_from_eng, "src-train.txt")
    from_eng_tgt_train_dir = os.path.join(out_dir_from_eng, "tgt-train.txt")
    from_eng_src_val_dir = os.path.join(out_dir_from_eng, "src-val.txt")
    from_eng_tgt_val_dir = os.path.join(out_dir_from_eng, "tgt-val.txt")

    with open(to_eng_src_train_dir, "w", encoding="utf8") as file_to_eng_src_train:
        with open(to_eng_tgt_train_dir, "w", encoding="utf8") as file_to_eng_tgt_train:
            with open(to_eng_src_val_dir, "w", encoding="utf8") as file_to_eng_train_val:
                with open(to_eng_tgt_val_dir, "w", encoding="utf8") as file_to_eng_tgt_val:

                    with open(from_eng_src_train_dir, "w", encoding="utf8") as f_src_from_eng:
                        with open(from_eng_tgt_train_dir, "w", encoding="utf8") as f_trg_from_eng:
                            with open(from_eng_src_val_dir, "w", encoding="utf8") as f_src_from_eng_val:
                                with open(from_eng_tgt_val_dir, "w", encoding="utf8") as f_trg_from_eng_val:
                                    pbar = tqdm(range(num_of_batch))
                                    for i in pbar:
                                        pbar.set_description(f"# of sentences written: {i}")
                                        random_bibles = random.choices(list(bibles.keys()), k=bs)
                                        for random_bible in random_bibles:
                                            random_sentence_id = random.choice(list(target_bible.sentences.keys()))
                                            try:
                                                random_sentence_eng_bible = target_bible.sentences[random_sentence_id]
                                                random_sentence_random_bible = bibles[random_bible].sentences[random_sentence_id]
                                            except Exception as e:
                                                continue
                                            if len(random_sentence_eng_bible) < 10 or len(random_sentence_random_bible) < 10:
                                                continue
                                            if (num_of_batch - i) * bs > 50001:
                                                file_to_eng_src_train.write("<" + random_bible + "> " + random_sentence_random_bible + "\n")
                                                file_to_eng_tgt_train.write(random_sentence_eng_bible + "\n")

                                                f_src_from_eng.write(random_sentence_eng_bible + "\n")
                                                f_trg_from_eng.write("<" + random_bible + "> " + random_sentence_random_bible + "\n")
                                            else:
                                                file_to_eng_train_val.write("<" + random_bible + "> " + random_sentence_random_bible + "\n")
                                                file_to_eng_tgt_val.write(random_sentence_eng_bible + "\n")

                                                f_src_from_eng_val.write(random_sentence_eng_bible + "\n")
                                                f_trg_from_eng_val.write("<" + random_bible + "> " + random_sentence_random_bible + "\n")

    print("training/valid files are saved!:", out_dir)
    # create_valid(to_eng_src_train_dir, to_eng_tgt_train_dir, out_dir_to_eng, 50000)
    # create_valid(from_eng_src_train_dir, from_eng_tgt_train_dir, out_dir_from_eng, 50000)


def create_valid(f1, f2, out, valid_size):
    file1 = [line.replace("\n", "") for line in open(f1, "r", encoding="utf8").readlines()]
    file2 = [line.replace("\n", "") for line in open(f2, "r", encoding="utf8").readlines()]
    assert len(file1) == len(file2)
    random_sentences = sorted(random.choices(range(0, len(file1) - 1), k=valid_size), reverse=True)

    with open(out + "/src-valid.txt", "w", encoding="utf8") as f_valid_src:
        with open(out + "/tgt-valid.txt", "w", encoding="utf8") as f_valid_trg:
            for idx in random_sentences:
                f_valid_src.write(file1.pop(idx) + "\n")
                f_valid_trg.write(file2.pop(idx) + "\n")
    assert len(file1) == len(file2)

    with open(out + "/src-train.txt", "w", encoding="utf8") as f_train_src:
        with open(out + "/tgt-train.txt", "w", encoding="utf8") as f_train_trg:
            for idx in range(len(file1)):
                f_train_src.write(file1[idx] + "\n")
                f_train_trg.write(file2[idx] + "\n")

    print("training/valid files are saved!:", out)
    f_train_src.close()
    f_valid_src.close()
    f_train_trg.close()
    f_valid_trg.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--bible_dir', type=str, default=config.corpus_path)
    parser.add_argument('--outdir', type=str, default=config.translation_input_path)
    parser.add_argument('--bible_list_file', type=str, default=config.bible_list_file)
    parser.add_argument('--tgt_bible', type=str, default="eng-x-bible-new2007", help="The bible translation to translate")
    parser.add_argument('--thread', type=int, default=multiprocessing.cpu_count())

    parser.add_argument('--limit', type=int, default=2000, help="# of translations. Mainly used for testing")

    args = parser.parse_args()

    bs = 100
    bible_list_file = args.bible_list_file
    bible_dir = args.bible_dir
    limit = args.limit
    tgt_bible_name = args.tgt_bible
    out_dir = args.outdir

    if os.path.exists(out_dir) and len(os.listdir(out_dir)) != 0:
        print("{} is not empty, assuming the translation data is already prepared.".format(out_dir))
        exit()

    s = time.time()
    bibles = {}
    bible_list = {os.path.join(bible_dir, trans.strip()) for trans in open(bible_list_file, "r", encoding="utf8").readlines()[0:limit]}
    print("Bible translations are being read.. this step may take several minutes")
    with Pool(processes=args.thread) as pool:
        bibles_tmp = pool.map(read_and_transliterate, bible_list)
        for x in bibles_tmp:
            bibles.update(x)

    num_of_sentences = sum([len(x.sentences) for x in bibles.values()])
    num_of_batch = int(num_of_sentences / bs)
    src_bible = bibles[tgt_bible_name]

    print("bibles read in {} sec".format(time.time() - s))
    print("num of sentences:", num_of_sentences)
    print("num of batches:", int(num_of_sentences / bs))

    create_nmt_data(bibles, out_dir, num_of_batch, src_bible, bs)
