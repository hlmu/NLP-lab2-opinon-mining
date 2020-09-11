import paddle.fluid as fluid
import paddle
import numpy as np

import sys
import os
import math
import csv

from paddle.fluid.initializer import NormalInitializer

from utils import str2bool, get_logger, get_entity, to_lodtensor, get_entity
from data import read_corpus, read_dictionary, tag2label, random_embedding, data_reader, sentence2id
from train import get_vocab
from config import TEST_REVIEW, TEST_TEMP_OUTPUT

vocab_path = './data/word2id.pkl'
model_path = './model_/'

def test(model_num, vocab, use_gpu):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    temp_model_path = model_path + 'epoch' + str(model_num) + '/'
    print('model:'+temp_model_path)

    test_results = {}
    inference_scope = fluid.core.Scope()
    with open(TEST_REVIEW, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for row in reader:
            idx = int(row[0])
            sent = row[1]
            test_results[idx] = sent

    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
            fetch_targets] = fluid.io.load_inference_model(temp_model_path, exe)
        
        for idx, sent in test_results.items():
            wids_list = [sentence2id(sent, vocab)]
            result_list = exe.run(
                inference_program,
                feed={"word":to_lodtensor(wids_list, place)},
                fetch_list=fetch_targets,
                return_numpy=False
            )
            res = np.array(result_list[0])
            label2tag = {}
            for tag, label in tag2label.items():
                label2tag[label] = tag if label != 0 else label
            tag_list = [label2tag[label[0]] for label in res]
            test_results[idx] = (sent, tag_list)
        
        with open(TEST_TEMP_OUTPUT, 'w+', encoding='utf-8') as f:
            for idx,(sent, tag_list) in test_results.items():
                f.write(str(idx) + ',' + sent + ',' + str(tag_list) + '\n')

if __name__ == '__main__':
    vocab = get_vocab(vocab_path)
    test(99, vocab, False)