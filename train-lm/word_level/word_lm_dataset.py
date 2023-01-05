import pickle
import random
import numpy as np
import os


class word_lm_dataset:

    def __init__(self, lookup_dir, binary_dir, batch_size=100, constant=1, dim=300, limit=None):
        self.lookup_dir = lookup_dir
        self.binary_dir = binary_dir + "/%s.npy"
        self.constant = constant
        self.batch_size = batch_size
        self.dim = dim

        self.lookup_tables = {}
        self.id_trans_map = {}
        self.trans_id_map = {}

        self.trans_id_map["<pad_lang>"] = 0
        self.id_trans_map[0] = "<pad_lang>"

        for f in os.listdir(self.lookup_dir):
            translation = f.split(".")[0]
            lookup_table = pickle.load(open(os.path.join(self.lookup_dir, f), 'rb'))
            if len(lookup_table) == 0:
                print("empty npy file", f)
                continue
            self.lookup_tables[translation] = lookup_table
            self.id_trans_map[len(self.id_trans_map)] = translation
            self.trans_id_map[translation] = len(self.trans_id_map)
            if limit is not None and len(self.trans_id_map) > limit:  ## for testing
                break

        self.total_number_of_sentences = np.sum([len(x) for x in self.lookup_tables.values()])
        self.total_number_of_words = np.sum([tp[1] for x in self.lookup_tables.values() for tp in x])

        self.trans_id_map = self.trans_id_map
        self.number_of_languages = len(self.trans_id_map)

        self.pool = []
        print("The data is being prepared.")
        self.init_pool()
        self.print_statistics()

    def print_statistics(self):
        print("total number of sentences in the training data", self.total_number_of_sentences)
        print("total number of words in the training data", self.total_number_of_words)

        print("total number of languages in the training data", self.number_of_languages)
        print("total number of languages in the pool", len(set([x[0] for x in self.pool])))
        print("avg sent:", np.average([a[1] for x in self.lookup_tables.values() for a in x]))

    def init_pool(self):
        for translation, lookup_table in self.lookup_tables.items():
            selected_sentences = self.prepare_offsets(translation)
            if selected_sentences is not None:
                self.add_to_pool(translation, selected_sentences)

    def update_pool(self):
        trans_to_add = random.choice(list(self.lookup_tables.keys()))
        offset_to_add = self.prepare_offsets(trans_to_add)
        if offset_to_add is not None:
            self.add_to_pool(trans_to_add, offset_to_add)

    def add_to_pool(self, lang, offset):
        try:
            sent = np.memmap(self.binary_dir % lang, dtype='float32', mode='r', offset=offset[0],
                             shape=(np.sum(offset[1]), self.dim))  # ,shape=(1,300))
            start = 0
            for x in offset[1]:
                if x < 100:
                    self.pool.append((self.trans_id_map[lang], np.array(sent[start:start + x])))
                start += x
            del sent
        except FileNotFoundError as f:
            print("----", lang, f)
        except ValueError as f2:
            print("----", lang, f2)

    def prepare_offsets(self, translation):
        upper_bound = len(self.lookup_tables[translation]) - (self.batch_size + 1)
        if upper_bound > 0:  # if there are more sentences in that translation than the batch size
            first_sentence = random.randint(0, upper_bound)  # randomly selected first sentence
            seq_lengths = np.array([x[1] for x in self.lookup_tables[translation][first_sentence:first_sentence + self.batch_size]])
        else:
            first_sentence = 0
            seq_lengths = np.array([x[1] for x in self.lookup_tables[translation]])
        try:
            return (self.lookup_tables[translation][first_sentence][0], seq_lengths)
        except Exception as e:
            print("prepare exception", e, translation)
            return None

    def batch_generator(self):
        steps_per_epoch = int(self.total_number_of_sentences / self.batch_size) + 1
        for _ in range(steps_per_epoch):
            self.update_pool()
            random.shuffle(self.pool)
            batch = self.pool[0:self.batch_size]
            del self.pool[:self.batch_size]
            batch = self.process_batch(batch)
            yield batch

    def process_batch(self, batch):
        max_len = np.max([i[1].shape[0] for i in batch]) - 1

        seq_lengths = [i[1].shape[0] for i in batch]
        language_ids = np.array([t[0] for t in batch])
        language_ids_padded = np.zeros((len(batch), max_len))
        for idx, length in enumerate(seq_lengths):
            language_ids_padded[idx, :length] = language_ids[idx]

        x = [t[1][:-1] for t in batch]
        y = [t[1][1:] for t in batch]
        ## padding
        x = np.array([instance.tolist() + (max_len - len(instance)) * [np.zeros(300)] for instance in x])
        y = np.array([instance.tolist() + (max_len - len(instance)) * [np.zeros(300)] for instance in y])
        ##padding
        return ((x, language_ids_padded), y)


if __name__ == '__main__':
    #bin = "/home/murathan/Desktop/NEWLM/lm_input/np"
    #lookup = "/home/murathan/Desktop/NEWLM/lm_input/lookup"
    bin = "/home/murathan/PycharmProjects/final_language_embeddings/LM/word_level/prepare_input/lm_input/np"
    lookup = "/home/murathan/PycharmProjects/final_language_embeddings/LM/word_level/prepare_input/lm_input/lookup"

    dataset = word_lm_dataset(lookup, bin)

    for i, b in enumerate(dataset.batch_generator()):
        print(i)
