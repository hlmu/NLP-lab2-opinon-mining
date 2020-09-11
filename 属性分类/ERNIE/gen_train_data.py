from config import TRAIN_LABEL_CLEANED, TRAIN_DATASET, TEST_DATASET, DEV_DATASET
import csv

sents = []
with open(TRAIN_LABEL_CLEANED, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            feature = row[1] if row[1] != '_' else ''
            opinion = row[4] if row[4] != '_' else ''
            category = row[7]
            sents.append((feature + opinion, category))

sents_test = sents[:int(len(sents) * 0.15)]
sents_dev = sents[int(len(sents) * 0.15) : int(len(sents) * 0.3)]
sents_train = sents[int(len(sents) * 0.3):]

with open(TRAIN_DATASET, 'w+', encoding='utf-8') as f:
    for sent in sents_train:
        tmp_str = sent[0]
        category = sent[1]
        f.write(tmp_str + ',' + category + '\n')

with open(TEST_DATASET, 'w+', encoding='utf-8') as f:
    for sent in sents_test:
        tmp_str = sent[0]
        category = sent[1]
        f.write(tmp_str + ',' + category + '\n')

with open(DEV_DATASET, 'w+', encoding='utf-8') as f:
    for sent in sents_dev:
        tmp_str = sent[0]
        category = sent[1]
        f.write(tmp_str + ',' + category + '\n')