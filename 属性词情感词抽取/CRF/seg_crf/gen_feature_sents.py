# -*- coding: utf-8 -*-

"""
生成包含特征的句子分词文件，用以训练CRF
生成文件位于data/feature_sents_train.pkl和data/feature_sents_test.pkl
格式为：
[(idx1, sent_1), (idx2, sent_2), ...]
其中sent_n = [('词', '<词性>', (词在原句起始位置, 结束位置+1), (arc.head, arc.relation), '<标记>'), ]
其中<标记>包含：
    B-T 评价对象的开始单词
    I-T 评价对象除开始单词外的所有单词
    B-O 评价词语的开始单词
    I-O 评价词除开始单词外的所有单词
    OFF 其它词
训练数据包含<标记>，而测试数据不包含<标记>
例（训练数据）：
[(1, [('很', 'd', (0, 1), (2, 'ADV'), 'B-O'), 
    ('好', 'a', (1, 2), (0, 'HED'), 'I-O')])]
"""
import os
from pyltp import Postagger
from pyltp import Segmentor
from pyltp import Parser
from pyltp import NamedEntityRecognizer
import csv
import pickle
from config import LTP_DATA_DIR
from config import TRAIN_REVIEW, TRAIN_LABEL, FEATURE_SENTS_TRAIN, FEATURE_SENTS_TEST, TEST_REVIEW
import pkuseg

cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
sent_tags = {} # train_reviews 文件中的feature和opinion位置对句子编号建立的索引

segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型
postagger = Postagger() # 初始化实例
postagger.load(pos_model_path)  # 加载模型
parser = Parser() # 初始化实例
parser.load(par_model_path)  # 加载模型
seg = pkuseg.pkuseg(model_name='web')

# 对文本进行序列标注
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
            # words = segmentor.segment(row[1])  # 分词
            words = seg.cut(row[1])
            # print ('\t'.join(words))
            postags = postagger.postag(words)  # 词性标注
            # print ('\t'.join(postags))
            arcs = parser.parse(words, postags)  # 句法分析
            # print ("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
            netags = netagger(words, int(row[0]))
            # print ('\t'.join(netags))
            position = 0
            sent = []
            for idx, (word, pos, arc, netag) in enumerate(zip(words, postags, arcs, netags)):
                next_position = position + len(word)
                sent.append((word, pos, (position, next_position), (arc.head, arc.relation), netag))
                position = next_position
            sents_train.append((int(row[0]), sent))
        
        print(sents_train[0])

with open(TEST_REVIEW, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        sents_test = []
        for row in reader:
            row[1] = row[1].replace(' ', '，')  # 处理掉行中空格，CRF的F1值增加2.6个百分点
            # words = segmentor.segment(row[1])  # 分词
            words = seg.cut(row[1])
            # print ('\t'.join(words))
            postags = postagger.postag(words)  # 词性标注
            # print ('\t'.join(postags))
            arcs = parser.parse(words, postags)  # 句法分析
            # print ("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
            position = 0
            sent = []
            for idx, (word, pos, arc) in enumerate(zip(words, postags, arcs)):
                next_position = position + len(word)
                sent.append((word, pos, (position, next_position), (arc.head, arc.relation)))
                position = next_position
            sents_test.append((int(row[0]), sent))
        
        print(sents_test[0])

segmentor.release()  # 释放模型
postagger.release()  # 释放模型
parser.release()  # 释放模型

with open(FEATURE_SENTS_TRAIN, 'wb') as f:
    pickle.dump(sents_train, f)

with open(FEATURE_SENTS_TEST, 'wb') as f:
    pickle.dump(sents_test, f)