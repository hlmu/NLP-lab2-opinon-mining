#coding:utf-8

from collections import namedtuple
import codecs
import csv
from config import CATEGORIES, TRAIN_DATASET, DEV_DATASET, TEST_DATASET

from paddlehub.dataset import InputExample, HubDataset

class MyData(HubDataset):
    def __init__(self):
        self._load_train_examples()
        self._load_test_examples()
        self._load_dev_examples()

    def _load_train_examples(self):
        self.train_examples = self._read_tsv(TRAIN_DATASET)

    def _load_dev_examples(self):
        self.dev_examples = self._read_tsv(DEV_DATASET)

    def _load_test_examples(self):
        self.test_examples = self._read_tsv(TEST_DATASET)

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_labels(self):
        return CATEGORIES

    @property
    def num_labels(self):
        return len(self.get_labels())

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with codecs.open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f)
            examples = []
            seq_id = 0
            for line in reader:
                example = InputExample(
                    guid=seq_id, label=line[1], text_a=line[0])
                seq_id += 1
                examples.append(example)
            return examples


if __name__ == "__main__":
    ds = MyData()
    for e in ds.get_train_examples():
        print("{}\t{}\t{}\t{}".format(e.guid, e.text_a, e.text_b, e.label))