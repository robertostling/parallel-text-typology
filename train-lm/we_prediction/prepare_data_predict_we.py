import pickle
import random
from random import randint

import numpy as np


class we_prediction_dataset:
    def __init__(self, we_predict_pickle, char_set_pickle, batch_size=100, valid_size=25):
        self.valid_size = valid_size
        self.batch_size = batch_size
        self.data = pickle.load(open(we_predict_pickle, "rb"))

        self.trans_id_map = {k: v + 1 for v, k in enumerate(list(self.data.keys()))}
        self.id_trans_map = {v + 1: k for v, k in enumerate(list(self.data.keys()))}
        self.id_trans_map[0] = "<pad_lang>"

        self.num_of_lang = len(self.id_trans_map)
        try:
            with open("pickles/predict_we_id_lang_map.pickle", "wb") as f:
                pickle.dump(self.id_trans_map, f)
        except:
            print("dumping id_trans_map failed")

        self.char_set = sorted(pickle.load(open(char_set_pickle, "rb")))
        self.char_set.append("<")
        self.char_set.append(">")
        self.char_set = sorted(self.char_set)
        self.char_to_id = {k: v + 1 for v, k in enumerate(self.char_set)}
        self.id_to_char = {v + 1: k for v, k in enumerate(self.char_set)}
        self.id_to_char[0] = "<pad>"

        self.char_size = len(self.id_to_char)

        corpus_size = np.sum([len(x) for _, x in self.data.items()])

        print("# of words in training:", corpus_size)
        self.validation_data = self.create_valid()
        self.corpus_size = np.sum([len(x) for _, x in self.data.items()])
        print("# of words in training:", self.corpus_size)
        print("# of words in validation:", len(self.validation_data[0][0]))
        print("# of languages", self.num_of_lang)

    def create_valid(self):
        valid_data = []
        for bible, list_of_words in self.data.items():
            if len(list_of_words) / 2 < self.valid_size:
                continue
            for p in range(self.valid_size):
                word_pair = list_of_words.pop(randint(0, len(list_of_words) - 1))
                char_list = [self.char_to_id[c] for c in "<" + word_pair[0] + ">"]
                valid_data.append(((self.trans_id_map[bible], char_list), word_pair[1]))
        return self.process_batch(valid_data)

    def batch_generator(self):
        while True:
            random_bibles = random.choices(list(self.trans_id_map.keys()), k=self.batch_size)
            batch = []
            for bible in random_bibles:
                random_example = random.choice(self.data[bible])
                word = "<" + random_example[0] + ">"
                word = [self.char_to_id[c] for c in word]
                batch.append(((self.trans_id_map[bible], word), random_example[1]))
            batch = self.process_batch(batch)
            yield batch

    def process_batch(self, batch):
        try:
            max_len = np.max([len(pair[0][1]) for pair in batch])
        except ValueError:
            print("valueerror", batch)
        char_list = [t[0][1] for t in batch]
        target_embed = np.array([t[1] for t in batch])

        language_ids = np.array([t[0][0] for t in batch])
        language_ids = [np.repeat(language_ids[index], len(char_list[index])) for index in range(len(char_list))]

        ##padding
        char_list = np.array([instance + (max_len - len(instance)) * [0] for instance in char_list])
        language_ids = np.array([instance.tolist() + (max_len - len(instance)) * [0] for instance in language_ids])
        ##padding
        target_embed = np.array([instance.tolist() + (max_len - len(instance)) * np.zeros(300) for instance in target_embed])
        return ((char_list, language_ids), target_embed)


if __name__ == "__main__":

    d = we_prediction_dataset("pickle/we_prediction_data_small.pickle",
                              "/home/murathan/PycharmProjects/tf_language_embeddings/pickles/char_set.pickle")
    for x in d.batch_generator():
        print(len(x))
