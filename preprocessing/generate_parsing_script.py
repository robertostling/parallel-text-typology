import os.path
import json
from collections import defaultdict

from mulres.config import turku_model_path as model_path
from mulres.utils import load_resources_table

my_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

def main():
    text_table = load_resources_table()

    for info in text_table.values():
        if 'ud' in info and info.get('preferred_source') == 'yes':
            model = os.path.join(
                    model_path, 'models_' + info['ud'], 'pipelines.yaml')
            in_file = os.path.join(my_path, 'data', 'resources',
                    'parallel-text-toparse', info['name']+'.turku')
            out_file = os.path.join(my_path, 'data', 'processed',
                    'ud', info['name']+'.conllu')
            print('python3 full_pipeline_stream.py '
                  '--gpu -1 --conf {} <{} >{}'.format(model, in_file, out_file))
            #print('python3 full_pipeline_stream.py --gpu -1 --conf /hd1/turkunlp/models/models_{}/pipelines.yaml parse_plaintext <~/projects/multilingual-resources/data/resources/parallel-text-toparse/{}.turku >~/projects/multilingual-resources/processed/ud/{}.conllu'.format(info['ud'], info['name'], info['name']))

if __name__ == '__main__': main()

