# -*- coding: utf-8 -*-

"""
"""
import os
import csv
from config import LTP_DATA_DIR
from config import TRAIN_REVIEW, TRAIN_LABEL, FEATURE_SENTS_TRAIN, FEATURE_SENTS_TEST, TEST_REVIEW, LEXICON, CORPUS_TRAIN \
    ,EDA_REVIEW, EDA_LABEL
sent_tags = {}
# train_reviews 文件中的feature和opinion位置对句子编号建立的索引

# 对文本进行序列标注
def netagger(sent, sent_id):
    ne = ['O'] * len(sent)
    tags = sent_tags[sent_id]
    for tag in tags:
        start_pos = tag[0]
        end_pos = tag[1]
        category = tag[2]
        first = False
        for idx in range(len(sent)):
            if idx >= start_pos and idx < end_pos:
                if not first:
                    ne[idx] = 'B-T' if category == 'feature' else 'B-O'
                    first = True
                else:
                    ne[idx] = 'I-T' if category == 'feature' else 'I-O'
    return ne

# 5,_, , ,很是划算,3,7,价格,正面
# 6,香味,14,16,淡淡的,11,14,气味,正面
with open(TRAIN_LABEL, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            idx = int(row[0])
            if idx not in sent_tags:
                sent_tags[idx] = []
            if row[1] != '_':
                sent_tags[idx].append((int(row[2]), int(row[3]), 'feature'))
            if row[4] != '_':
                sent_tags[idx].append((int(row[5]), int(row[6]), 'opinion'))

with open(TRAIN_REVIEW, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        sents_train = []
        for row in reader:
            row[1] = row[1].replace(' ', '，')  # 处理掉行中空格，CRF的F1值增加2.6个百分点
            sent_id = int(row[0])
            sent = row[1]
            netags = netagger(sent, sent_id)
            sents_train.append((sent_id, [(char, tag) for char, tag in zip(sent, netags)]))        
        print(sents_train[0])

with open(CORPUS_TRAIN, 'w+', encoding='utf-8') as f:
    for sent in sents_train:
        for tmp_sent in sent[1]:
            f.write(tmp_sent[0] + ' ' + tmp_sent[-1])
            f.write('\n')
        f.write('\n')

sents_train = sents_train[:int(len(sents_train) * 0.9)]
sents_test = sents_train[int(len(sents_train) * 0.9):]

with open(FEATURE_SENTS_TRAIN, 'w+', encoding='utf-8') as f:
    for sent in sents_train:
        for tmp_sent in sent[1]:
            f.write(tmp_sent[0] + ' ' + tmp_sent[-1])
            f.write('\n')
        f.write('\n')

with open(FEATURE_SENTS_TEST, 'w+', encoding='utf-8') as f:
    for sent in sents_test:
        for tmp_sent in sent[1]:
            f.write(tmp_sent[0] + ' ' + tmp_sent[-1])
            f.write('\n')
        f.write('\n')