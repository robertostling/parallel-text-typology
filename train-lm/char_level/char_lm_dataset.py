import os, json, gzip, random
from collections import defaultdict
from os import listdir
import pickle
import numpy as np
import tensorflow as tf
import copy


class char_lm_dataset:
    def __init__(self, bible_dir, resource_file, batch_size=100, limit=None, valid_size=10, sentence_len=256):
        self.batch_size = batch_size
        self.sentence_len = sentence_len
        self.valid_size = valid_size
        self.dir = bible_dir
        self.translations = []
        self.corpus = defaultdict(str)
        rsc = json.load(open(resource_file))
        allowed_translations = [translation for translation, values in rsc.items() if values["preferred_source"] != "yes"]
        print("Preparing data....")
        self.char_set = set()  # {'N', 'S', 'X', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}
        for i, f in enumerate(sorted(listdir(self.dir))):
            if "lif-x-bible-2009" in f: continue
            if ".DS" in f: continue
            if limit and i >= limit: break
            if f.split(".")[0] not in allowed_translations: continue  # skip the source translations
            trans = f.split(".")[0]
            self.translations.append(trans)
            sentences = [sentence.decode("utf8").replace("\n", "") for sentence in gzip.open(os.path.join(self.dir, f), "r") if
                         len(sentence.decode("utf8").replace("\n", "")) > 1]
            self.char_set.update("".join(sentences))
            if len(sentences) > 100:
                self.corpus[trans] = " ".join(sentences)
            else:
                print("There is not enough sentences in", f, len(sentences))
        print("The corpus has been read!")
        self.trans_id_map = {k: v + 1 for v, k in enumerate(self.corpus.keys())}
        self.id_trans_map = {v + 1: k for v, k in enumerate(self.corpus.keys())}
        self.id_trans_map[0] = "<pad_lang>"

        self.char_set.add(" ")
        self.char_set = sorted(self.char_set)

        self.char_to_id = {k: v + 1 for v, k in enumerate(self.char_set)}
        self.id_to_char = {v + 1: k for v, k in enumerate(self.char_set)}
        self.id_to_char[0] = "<pad>"
        self.char_size = len(self.id_to_char)

        self.validation_data = self.create_valid()
        bible_len = [len(bib) for bib in self.corpus.values()]

        self.corpus_size = int(sum([len(x) for x in self.corpus.values()]) / self.sentence_len)
        self.num_of_lang = len(self.id_trans_map)
        print("-" * 15)
        print("Char set", sorted(list(self.char_set)))
        print("# of translations in the corpus", len(self.corpus))
        print("longest bible: {} chars".format(max(bible_len)))
        print("shortest bible: {} chars".format(min(bible_len)))
        print("average sentence per bible (assuming sentence len= {}): {}".format(self.sentence_len, np.average([bib / self.sentence_len for bib in bible_len])))
        print("language count", self.num_of_lang)
        print("sentence count", self.corpus_size)
        print("-" * 15)

    def create_valid(self):
        valid_data = []
        for bible, list_of_sentences in self.corpus.items():
            for p in range(self.valid_size):
                start_index = random.randint(0, (len(self.corpus[bible]) - (self.sentence_len + 1)))
                sentence = [self.char_to_id[c] for c in self.corpus[bible][start_index:start_index + self.sentence_len + 1]]  # pick a random chunk
                self.corpus[bible] = self.corpus[bible][:start_index] + " " + self.corpus[bible][start_index + self.sentence_len + 1:]
                valid_data.append((self.trans_id_map[bible], sentence))
        return self.process_batch(valid_data)

    def batch_generator(self):
        total_steps = int(self.corpus_size / self.batch_size) + 1
        for _ in range(total_steps):
            random_bibles = random.choices(list(self.trans_id_map.keys()), k=self.batch_size)
            batch = []
            for bible in random_bibles:
                # sentence = random.choice(self.corpus[bible])
                start_index = random.randint(0, (len(self.corpus[bible]) - (self.sentence_len + 1)))
                sentence = [self.char_to_id[c] for c in self.corpus[bible][start_index:start_index + self.sentence_len + 1]]  # pick a random sentence
                batch.append((self.trans_id_map[bible], sentence))
            batch = self.process_batch(batch)
            yield batch

    def process_batch(self, batch):
        x = np.array([t[1][:-1] for t in batch])
        y = [t[1][1:] for t in batch]
        y = np.array([tf.keras.utils.to_categorical(c, self.char_size) for c in y])

        language_ids = np.array([i[0] for i in batch])
        language_ids = np.array([np.repeat(language_ids[index], len(x[index])) for index in range(len(x))])

        return (x, language_ids), y

class char_lm_dataset_word:
    def __init__(self, wordlist_pickle, char_set_pickle, batch_size = 100, valid_size = 100):
        self.valid_size = valid_size
        self.batch_size = batch_size
        self.dir = wordlist_pickle
        self.wordlist = pickle.load(open(wordlist_pickle, "rb"))

        self.trans_id_map = {k: v + 1 for v, k in enumerate(sorted(list(self.wordlist.keys())))}
        self.id_trans_map = {v + 1:k for v, k in enumerate(sorted(list(self.wordlist.keys())))}
        self.id_trans_map[0] = "<pad_lang>"

        self.num_of_lang = len(self.id_trans_map)

        with open("pickles/char_id_lang_map.pickle", "wb") as f:
            pickle.dump(self.id_trans_map, f)

        self.char_set = sorted(pickle.load(open(char_set_pickle, "rb")))
        self.char_set.append("<")
        self.char_set.append(">")
        self.char_set = sorted(self.char_set)
        self.char_to_id = {k: v+1 for v, k in enumerate(self.char_set)}
        self.id_to_char = {v+1: k for v, k in enumerate(self.char_set)}
        self.id_to_char[0] = "<pad>"

        self.char_size = len(self.id_to_char)

        #print("# of words:", np.sum([len(x) for _, x in self.wordlist.items()]))

        self.prune_long_words()

        print("# of words:", np.sum([len(x) for _, x in self.wordlist.items()]))
        print("longest word:",np.max([len(word) for _, x in self.wordlist.items() for word in x]))
        print("shortest word:", np.min([len(word) for _, x in self.wordlist.items() for word in x]))
        print("average word:",np.average([len(word) for _, x in self.wordlist.items() for word in x]))

        self.validation_data = self.create_valid()
        self.corpus_size = np.sum([len(x) for _, x in self.wordlist.items()])
        print("# of words in training:", self.corpus_size)
        print("# of words in validation:", len(self.validation_data[0][0]))
        print("# of languages", self.num_of_lang)

    def prune_long_words(self, limit=100):
        pruned_word_list = copy.deepcopy(self.wordlist)
        long_word_dict = defaultdict(int)
        for trans, wl in self.wordlist.items():
            for word in wl:
                if len(word) > limit:
                    pruned_word_list[trans].remove(word)
                    long_word_dict[trans] +=1

        self.wordlist = pruned_word_list
        #print("pruned words:", long_word_dict, "total", sum([v for v in long_word_dict.values()]))

    def create_valid(self):
        valid_data = []
        for bible, list_of_words in self.wordlist.items():
            for p in range(self.valid_size):
                word = "<" + list_of_words.pop(random.randint(0, len(list_of_words)-1)) + ">"
                word = [self.char_to_id[c] for c in word]
                valid_data.append((self.trans_id_map[bible], word))
        return self.process_batch(valid_data)

    def batch_generator(self):
        while True:
            random_bibles = random.sample(list(self.trans_id_map.keys()), self.batch_size)
            batch = []
            for bible in random_bibles:
                word = "<" + random.choice(self.wordlist[bible]) + ">"
                word = [self.char_to_id[c] for c in word]
                batch.append((self.trans_id_map[bible], word))
            batch = self.process_batch(batch)
            yield batch

    def process_batch(self, batch):
        try:
            max_len = np.max([len(pair[1]) for pair in batch]) - 1
        except ValueError:
            print("valueerror", batch)
        x = [t[1][:-1] for t in batch]
        y = [t[1][1:] for t in batch]

        language_ids = np.array([i[0] for i in batch])
        language_ids = [ np.repeat(language_ids[index], len(x[index]))  for index in range(len(x))]

        ##padding
        x = np.array([instance + (max_len - len(instance)) * [0] for instance in x])
        language_ids = np.array([instance.tolist() + (max_len - len(instance)) * [0] for instance in language_ids])
        y =  np.array([instance + (max_len - len(instance)) * [0] for instance in y])
        ##padding
        y = np.array([tf.keras.utils.to_categorical(c, self.char_size) for c in y])
        #print(max_len)
        return ((x, language_ids), y)

if __name__ == "__main__":
    x = char_lm_dataset("/home/murathan/Desktop/NEWLM/transliterated", resource_file="/home/murathan/Desktop/NEWLM/resources.json", limit=5000, batch_size=4)
    for a in x.batch_generator():
        print(a[0][0].shape)
        print(a[1].shape)
