#coding:utf-8
#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import codecs
import csv
import json
from collections import namedtuple

from config import TRAIN_DATASET, TEST_DATASET, DEV_DATASET
from paddlehub.dataset import InputExample

tag2label = {"O": 0,
             "B-T": 1, "I-T": 2,
             "B-O": 3, "I-O": 4
             }


class My_DATA():
    """
    A set of manually annotated Chinese word-segmentation data and
    specifications for training and testing a Chinese word-segmentation system
    for research purposes.  For more information please refer to
    https://www.microsoft.com/en-us/download/details.aspx?id=52531
    """

    def __init__(self):
        self._load_train_examples()
        self._load_test_examples()
        self._load_dev_examples()

    def _load_train_examples(self):
        train_file = os.path.join(TRAIN_DATASET)
        self.train_examples = self._read_tsv(train_file)

    def _load_dev_examples(self):
        self.dev_file = os.path.join(DEV_DATASET)
        self.dev_examples = self._read_tsv(self.dev_file)

    def _load_test_examples(self):
        self.test_file = os.path.join(TEST_DATASET)
        self.test_examples = self._read_tsv(self.test_file)

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_labels(self):
        return ["B-T", "I-T", "B-O", "I-O", "O"]

    @property
    def num_labels(self):
        """
        Return the number of labels in the dataset.
        """
        return len(self.get_labels())

    def get_label_map(self):
        return tag2label

    def _read_tsv(self, input_file):
        """Reads a tab separated value file."""
        data = []
        seq_id = 0
        with open(input_file, encoding='utf-8') as fr:
            lines = fr.readlines()
        sent_, tag_ = [], []
        for line in lines:
            if line != '\n':
                [char, label] = line.strip().split()
                sent_.append(char)
                tag_.append(label)
            else:
                tmp_data = InputExample(guid=seq_id, label='\x02'.join(tag_), text_a='\x02'.join(sent_))
                data.append(tmp_data)
                seq_id += 1
                sent_, tag_ = [], []
        return data

# if __name__ == "__main__":
#     ds = My_DATA()
#     count = 0
#     for e in ds.get_train_examples():
#         count+= 1
#         if count <= 10:
#             print("{}\t{}\t{}\t{}".format(e.guid, e.text_a, e.text_b, e.label))
#         else:
#             break