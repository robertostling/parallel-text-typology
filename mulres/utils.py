import json
import os.path

import mulres.config

def load_ids_pos_table():
    pos_ids = {
            'ADJ': ['4-810','12-310','12-550','12-560','12-570','14-130',
                    '14-140','14-150','16-710','16-720','16-810'],
            'NUM': ['13-20','13-30','13-40','13-50','13-60','13-70','13-80',
                    '13-90','13-100']
            }
    ids_pos = {}
    for pos, ids_list in pos_ids.items():
        for ids in ids_list:
            ids_pos[ids] = pos
    return ids_pos


def load_resources_table():
    with open(mulres.config.json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_verse_table():
    with open(mulres.config.verses_path, 'r') as f:
        return f.read().split()

def load_contact_graph():
    neighbors = {}
    with open(mulres.config.contact_path, 'r') as f:
        for line in f:
            codes = line.split()
            code_set = set(codes)
            for code in codes:
                neighbors[code] = neighbors.get(code, set()) | code_set
    return neighbors

def encoded_filename(name):
    return os.path.join(mulres.config.encoded_path, name+'.gz')

def aligned_filename(name):
    return os.path.join(mulres.config.aligned_path, name+'.gz')

def wordorder_filename(name):
    return os.path.join(mulres.config.wordorder_path, name+'.tab')

def embeddings_filename(name):
    return os.path.join(mulres.config.embeddings_path, name+'.vec')

def transliterated_filename(name):
    return os.path.join(mulres.config.transliterated_path, name+'.txt.gz')

def paradigms_filename(name):
    return os.path.join(mulres.config.paradigms_path, name+'.txt.gz')


def cached_embeddings_filename(name):
    return os.path.join(mulres.config.embeddings_cache_path, name+'.vec')

