from config import TRAIN_LABEL_CLEANED, TRAIN_REVIEW_CLEANED, TRAIN_DATASET, TEST_DATASET, DEV_DATASET
import csv
import random

sents = []
sents_set = set()
sents_raw = {}
# sents_raw = {}
sent_features = {}
sent_opinions = {}
all_features = set()
all_opinions = set()

cnt1 = 0
with open(TRAIN_REVIEW_CLEANED, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        idx = int(row[0])
        sent = row[1]
        sents_raw[idx] = sent

with open(TRAIN_LABEL_CLEANED, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            idx = int(row[0])
            feature = row[1] if row[1] != '_' else ''
            opinion = row[4] if row[4] != '_' else ''
            if not feature or not opinion:
                continue
            sents.append((sents_raw[idx], feature+'|'+opinion, '1'))
            sents_set.add((sents_raw[idx], feature+'|'+opinion))
            if idx not in sent_features:
                sent_features[idx] = set()
                sent_opinions[idx] = set()
            sent_features[idx].add(feature)
            sent_opinions[idx].add(opinion)
            all_features.add(feature)
            all_opinions.add(opinion)
            cnt1 += 1


cnt0 = 0
for key in sent_features:
    features = sent_features[key]
    opinions = sent_opinions[key]
    for feature in features:
        for opinion in opinions:
            if (sents_raw[key], feature+'|'+ opinion) not in sents_set:
                sents.append((sents_raw[key], feature+'|'+opinion, '0'))
                # sents_set.add((feature, opinion, '0'))
                cnt0 += 1

# complement_false = set()
# all_features = list(all_features)
# all_opinions = list(all_opinions)
# for feature in all_features:
#     for opinion in all_opinions:
#         if (feature, opinion) not in sents_set:
#             complement_false.add((feature, opinion, '0'))
#             sents_set.add((feature, opinion, '0'))
# complement_false = list(complement_false)
# random.Random(4).shuffle(complement_false)

# for item in complement_false:
#     sents.append(item)
#     cnt0 += 1
#     if cnt0 >= cnt1:
#         break

print('instances of type 1: ', cnt1)
print('instances of type 0: ', cnt0)

random.Random(4).shuffle(sents)

sents_test = sents[:int(len(sents) * 0.15)]
sents_dev = sents[int(len(sents) * 0.15) : int(len(sents) * 0.3)]
sents_train = sents[int(len(sents) * 0.3):]

with open(TRAIN_DATASET, 'w+', encoding='utf-8') as f:
    for sent in sents_train:
        feature = sent[0]
        opinion = sent[1]
        cla = sent[2]
        f.write(feature + ',' + opinion + ',' + cla + '\n')

with open(TEST_DATASET, 'w+', encoding='utf-8') as f:
    for sent in sents_test:
        feature = sent[0]
        opinion = sent[1]
        cla = sent[2]
        f.write(feature + ',' + opinion + ',' + cla + '\n')

with open(DEV_DATASET, 'w+', encoding='utf-8') as f:
    for sent in sents_dev:
        feature = sent[0]
        opinion = sent[1]
        cla = sent[2]
        f.write(feature + ',' + opinion + ',' + cla + '\n')