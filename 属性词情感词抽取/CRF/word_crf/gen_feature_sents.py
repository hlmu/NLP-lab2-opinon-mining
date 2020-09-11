# -*- coding: utf-8 -*-
"""
生成包含特征的句子分词文件，用以训练CRF
生成文件位于data/feature_sents_train.pkl和data/feature_sents_test.pkl
格式为：
[(idx1, sent_1), (idx2, sent_2), ...]
其中sent_n = [('字', (字在原句起始位置, 结束位置+1), '<标记>'), ]
其中<标记>包含：
    B-T 评价对象的开始的字
    I-T 评价对象除开始字外的所有字
    B-O 评价词语的开始字
    I-O 评价词除开始字外的所有字
    OFF 其它字
训练数据包含<标记>，而测试数据不包含<标记>
例（训练数据）：
[(1, [('很', (0, 1), 'B-O'), 
    ('好', (1, 2), 'I-O')])]
"""
import os
from pyltp import Postagger
from pyltp import Segmentor
from pyltp import Parser
from pyltp import NamedEntityRecognizer
import csv
import pickle
from config import TRAIN_REVIEW, TRAIN_LABEL, FEATURE_SENTS_TRAIN, FEATURE_SENTS_TEST, TEST_REVIEW, \
                MATCH_TEST_REVIEW, MATCH_TEST_LABEL, MATCH_FEATURE_SENTS_TEST
sent_tags = {} # train_reviews 文件中的feature和opinion位置对句子编号建立的索引

def q_to_b(q_str):
    """全角转半角"""
    b_str = ""
    for uchar in q_str:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65374 >= inside_code >= 65281:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        b_str += chr(inside_code)
    return b_str

def netagger(words, sent_id):
    ne = ['OFF'] * len(words)
    tags = sent_tags[sent_id]
    for tag in tags:
        start_pos = tag[0]
        end_pos = tag[1]
        category = tag[2]
        position = 0
        first = False
        for idx, word in enumerate(words):
            next_position = position + len(word)
            if position >= start_pos and next_position <= end_pos:
                if not first:
                    ne[idx] = 'B-T' if category == 'feature' else 'B-O'
                    first = True
                else:
                    ne[idx] = 'I-T' if category == 'feature' else 'I-O'

            position = next_position

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
            row[1] = q_to_b(row[1])
            words = list(row[1])
            netags = netagger(words, int(row[0]))
            position = 0
            sent = []
            for idx, (word, netag) in enumerate(zip(words, netags)):
                next_position = position + len(word)
                sent.append((word, (position, next_position), netag))
                position = next_position
            sents_train.append((int(row[0]), sent))
        
        print(sents_train[0])

with open(TEST_REVIEW, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        sents_test = []
        for row in reader:
            row[1] = row[1].replace(' ', '，')  # 处理掉行中空格，CRF的F1值增加2.6个百分点
            row[1] = q_to_b(row[1])
            words = list(row[1])
            netags = netagger(words, int(row[0]))
            position = 0
            sent = []
            for idx, (word, ) in enumerate(zip(words, )):
                next_position = position + len(word)
                sent.append((word, (position, next_position)))
                position = next_position
            sents_test.append((int(row[0]), sent))
        
        print(sents_test[0])

with open(MATCH_TEST_REVIEW, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        sents_match_test = []
        for row in reader:
            row[1] = row[1].replace(' ', '，')  # 处理掉行中空格，CRF的F1值增加2.6个百分点
            row[1] = q_to_b(row[1])
            words = list(row[1])
            netags = netagger(words, int(row[0]))
            position = 0
            sent = []
            for idx, (word, ) in enumerate(zip(words, )):
                next_position = position + len(word)
                sent.append((word, (position, next_position)))
                position = next_position
            sents_match_test.append((int(row[0]), sent))
        
        print(sents_match_test[0])

with open(FEATURE_SENTS_TRAIN, 'wb') as f:
    pickle.dump(sents_train, f)

with open(FEATURE_SENTS_TEST, 'wb') as f:
    pickle.dump(sents_test, f)

with open(MATCH_FEATURE_SENTS_TEST, 'wb') as f:
    pickle.dump(sents_match_test, f)