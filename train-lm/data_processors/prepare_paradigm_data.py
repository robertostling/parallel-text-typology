import argparse
import gzip
import os
import random
from collections import defaultdict

from mulres import config
from mulres import transliterate
from tqdm import tqdm

random.seed(42)


def write_to_file(pos_tag, pair_list, file_list):
    template_mixed = "<{}> <{}> {}"
    template = "<{}> {}"
    for pair in pair_list:
        translation = pair[0].split(".")[0]
        pair = pair[1:]
        file_list[pos_tag][0].write(template.format(translation.split(".")[0], " ".join(pair[0]) + "\n"))
        file_list[pos_tag][1].write(" ".join(pair[1]) + "\n")
        ### write to mixed file as well
        file_list["MIXED"][0].write(template_mixed.format(translation.split(".")[0], str(pos_tag), " ".join(pair[0]) + "\n"))
        file_list["MIXED"][1].write(" ".join(pair[1]) + "\n")


def prepare_file(paradigm_folder, out_dir, size, valid_ratio=0.95, overwrite=False):
    if os.path.exists(out_dir + "_noun") and not overwrite:
        print("{} is not empty, assuming the input is already prepared.".format(out_dir + "_noun"))
        return
    if not os.path.exists(out_dir + "_noun"): os.makedirs(out_dir + "_noun")
    if not os.path.exists(out_dir + "_verb"): os.makedirs(out_dir + "_verb")
    if not os.path.exists(out_dir + "_mixed"): os.makedirs(out_dir + "_mixed")

    train_files = {"NOUN": [open("%s_noun/src-train.txt" % out_dir, "w", encoding="utf8"), open("%s_noun/tgt-train.txt" % out_dir, "w", encoding="utf8")],
                   "VERB": [open("%s_verb/src-train.txt" % out_dir, "w", encoding="utf8"), open("%s_verb/tgt-train.txt" % out_dir, "w", encoding="utf8")],
                   "MIXED": [open("%s_mixed/src-train.txt" % out_dir, "w", encoding="utf8"), open("%s_mixed/tgt-train.txt" % out_dir, "w", encoding="utf8")]}
    valid_files = {"NOUN": [open("%s_noun/src-val.txt" % out_dir, "w", encoding="utf8"), open("%s_noun/tgt-val.txt" % out_dir, "w", encoding="utf8")],
                   "VERB": [open("%s_verb/src-val.txt" % out_dir, "w", encoding="utf8"), open("%s_verb/tgt-val.txt" % out_dir, "w", encoding="utf8")],
                   "MIXED": [open("%s_mixed/src-val.txt" % out_dir, "w", encoding="utf8"), open("%s_mixed/tgt-val.txt" % out_dir, "w", encoding="utf8")]}

    list_of_translations = {translation: [(line.strip().split(maxsplit=1)[0], transliterate.remove_distinctions(transliterate.normalize(line.strip().split(maxsplit=1)[1])).split())
                                          for line in gzip.open(os.path.join(paradigm_folder, translation), "rt", encoding="utf8")]
                            for translation in os.listdir(paradigm_folder)}

    print("files are read.")
    pair_dict = defaultdict(list)
    error = 0
    pbar = tqdm(range(size))
    for i in pbar:
        pbar.set_description(f"Paradigms created {i}")
        translation = random.choice(list(list_of_translations.keys()))
        pos_tag, paradigms = random.choice(list_of_translations[translation])
        if len(paradigms) == 0:
            error += 1
            continue
        elif len(paradigms) == 1:
            pair = [translation, paradigms[0], paradigms[0]]
        else:
            pair = random.choices(paradigms, k=2)
            pair.insert(0, translation)
        pair_dict[pos_tag].append(pair)
    for pos_tag, pair_list in pair_dict.items():
        train_pairs, valid_pairs = pair_list[:int(len(pair_list) * valid_ratio)], pair_list[int(len(pair_list) * valid_ratio):]
        valid_set = set(tuple(x) for x in valid_pairs)
        train_set = set(tuple(x) for x in train_pairs)
        valid_set_cleaned = valid_set - train_set
        print("{} {} common pairs are removed from the validation data".format(pos_tag, len(valid_set) - len(valid_set_cleaned)))
        write_to_file(pos_tag, train_pairs, train_files)
        write_to_file(pos_tag, valid_pairs, valid_files)
        print("{} {} paradigms have been saved".format(pos_tag, len(train_set)))
    print(error, "done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for Paradigm-based LEs')
    parser.add_argument('--paradigm_dir', type=str, default=config.paradigms_path)
    parser.add_argument('--out_dir', type=str, default=config.paradigms_input_path)
    parser.add_argument('--size', type=int, default=11800000)
    parser.add_argument('--overwrite', action="store_true")

    args = parser.parse_args()
    prepare_file(args.paradigm_dir, args.out_dir, args.size, overwrite=args.overwrite)
