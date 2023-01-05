import argparse
import json
import os
import pickle
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
import shutil
import numpy as np
from mulres.embeddings import Embeddings
from mulres.empfile import EncodedMPF
from mulres import config


def save_as_npy(pair, verse_count_dict, out_dir):
    embed = Embeddings(pair[0])
    bible = EncodedMPF(pair[1])
    converted, look_up = [], []
    skipped_sentences, oov_words = 0, 0
    embedding_dim = embed.dim
    first_pos = 0
    num_of_sentence = 0
    number_of_accepted_sentences = 0
    for index, sentence in enumerate(bible.sentences):
        if sentence is not None:
            num_of_sentence += 1
            verse = bible.verse_ids[index]
            sentence_surface = [bible.sentences_vocab[word_id] for word_id in sentence]
            sentence_embedded = [embed.embeddings[word]
                                 for word in sentence_surface if word in embed.embeddings.keys()]
            try:
                if len(sentence_embedded) / verse_count_dict[verse] < 0.8:
                    skipped_sentences += 1
                    continue
            except ZeroDivisionError:
                continue
            number_of_accepted_sentences += 1
            oov_words += (len(sentence_surface) - len(sentence_embedded))
            look_up.append((first_pos, len(sentence_embedded)))
            first_pos = first_pos + len(sentence_embedded) * embedding_dim * 4
            converted.extend(sentence_embedded)
    converted = np.array(converted)
    if number_of_accepted_sentences < 1000:
        print("The following translation with {}/{} remaining sentences will be omitted: {}".format(number_of_accepted_sentences, num_of_sentence, bible.name))
        return

    try:
        fp = np.memmap(os.path.join(out_dir[0], bible.name + ".npy"), dtype='float32', mode='w+', shape=(len(converted), embedding_dim))
        fp[:] = converted[:]
    except Exception as e:
        print(bible.name, e)
    with open(os.path.join(out_dir[1], bible.name + ".pickle"), "wb") as f:
        pickle.dump(look_up, f)

    print(
        "{:35s}: skipped sentences:  {}/{} ({:.2f}%) - OOV words in the remaning {} sentences: {}/{}".format(bible.name, skipped_sentences, number_of_accepted_sentences,
                                                                                                             skipped_sentences / len(bible.sentences) * 100,
                                                                                                             num_of_sentence,
                                                                                                             oov_words, len(converted)))


def compute_estimates(json_path, bible_dir):
    rsc = json.load(open(json_path))
    high_resource_translations = [translation for translation, values in rsc.items() if values["preferred_source"] == "yes"]
    verse_count_dict = defaultdict(list)
    content_pos = ['ADJ', 'ADV', 'NOUN', 'NUM', 'PROPN', 'VERB']
    for translation in high_resource_translations:
        bible = EncodedMPF(os.path.join(bible_dir, translation + ".gz"))
        if "pos" in bible.available_annotations:
            content_pos_id = [bible.pos_vocab.index(pos) for pos in content_pos if pos in bible.pos_vocab]
            for id, verse in enumerate(bible.verse_ids):
                if bible.sentences[id] is not None:
                    verse_count_dict[verse].append(sum([list(bible.pos[id]).count(pos_id) for pos_id in content_pos_id]))
    verse_count_dict = {k: np.average(v) for k, v in verse_count_dict.items()}
    pickle.dump(verse_count_dict, open("content_word_estimates.pickle", "wb"))
    print("done!")
    return verse_count_dict


def make_dir(dir_list, rewrite=True):
    if rewrite:
        print("Warning! The following directories will be deleted:\n", "\n".join(dir_list))
        for d in dir_list:
            if  os.path.exists(d): shutil.rmtree(d)
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-bible_dir', type=str, default=config.encoded_path, help="The directory of the ENCODED bibles")
    parser.add_argument('-embed_dir', type=str, default=config.embeddings_path, help="The directory of the projected word embeddings")
    parser.add_argument('-resource_file', type=str, default=config.json_path, help="The directory of the resources.json")

    parser.add_argument('--rewrite', action="store_true")
    parser.add_argument('-embed_size', type=int, default=300)
    parser.add_argument('-thread', type=int, default=12)
    parser.add_argument('-cache', type=int, default=100)

    args = parser.parse_args()
    args.outpath = config.word_lm_input_path
    if not os.path.exists("content_word_estimates.pickle"):
        print("first run, number of content words per verse is being estimated.")
        verse_count_dict = compute_estimates(args.resource_file, args.bible_dir)
    else:
        verse_count_dict = pickle.load(open("content_word_estimates.pickle", "rb"))
        print("estimated number of content words per verse is loaded")

    np_out_dir = os.path.join(args.outpath, "np")
    lookup_out_dir = os.path.join(args.outpath, "lookup")
    make_dir([args.outpath, np_out_dir, lookup_out_dir], args.rewrite)

    bible_dir = args.bible_dir
    projected_embedding_dir = [os.path.join(args.embed_dir, translation) for translation in sorted(os.listdir(args.embed_dir))]
    bible_file_dirs = [os.path.join(args.bible_dir, translation.replace("vec", "gz")) for translation in sorted(os.listdir(args.embed_dir))]
    to_be_converted = []
    start = time.time()
    for i, embed_dir in enumerate(projected_embedding_dir):
        # if sum([x in embed_dir for x in ["lzh", "mya", "khm","ksw"]]) == 0:continue
        to_be_converted.append((embed_dir, bible_file_dirs[i]))
        if len(to_be_converted) == args.cache:
            with Pool(processes=args.thread) as pool:
                pool.map(partial(save_as_npy, out_dir=(np_out_dir, lookup_out_dir), verse_count_dict=verse_count_dict), to_be_converted)
            to_be_converted = []
            duration = time.time() - start
            print("\t\t\t{} bibles took {} seconds ({:.2f} bib/sec)".format(args.cache, duration, args.cache / duration))
            start = time.time()
    with Pool(processes=args.thread) as pool:
        pool.map(partial(save_as_npy, out_dir=(np_out_dir, lookup_out_dir), verse_count_dict=verse_count_dict), to_be_converted)
