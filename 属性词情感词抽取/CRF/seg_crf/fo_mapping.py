"""
使用训练好的CRF属性词特征词标记模型为训练集打上标记，
并使用栈对属性词特征词进行一一对应，
结果位于results/task1_answer.csv
由于统计出给定训练集中跨标点的属性词特征词对占比<0.01，
这里假设所有相关的观点词属性词位于一个短句内（即属性词观点词之间不存在标点或分隔），
目前精度0.75
"""
import pickle
import sklearn
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from config import FEATURE_SENTS_TRAIN, FEATURE_SENTS_TEST, FOR_MODEL, TASK1_ANS
from utils.feature_extraction import sent2features, sent2labels, sent2index

with open(FEATURE_SENTS_TRAIN, 'rb') as f:
    train_sents = pickle.load(f)

with open(FEATURE_SENTS_TEST, 'rb') as f:
    test_sents = pickle.load(f)

with open(FOR_MODEL, 'rb') as f:
    crf = pickle.load(f)

# X = [sent2features(s) for s in train_sents]
# y = [sent2labels(s) for s in train_sents]
# indices = [sent2index(s) for s in train_sents]

# labels = list(crf.classes_) 
# labels.remove('OFF')
# X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
#     X, y, indices, test_size=0.1, shuffle=False)
# y_pred = crf.predict(X_test)
# print(metrics.flat_f1_score(y_test, y_pred,
#                       average='weighted', labels=labels))

X = [sent2features(s) for s in test_sents]
indices = [sent2index(s) for s in test_sents]
X_test = X
idx_test = indices
y_pred = crf.predict(X_test)

fo_result = []

for i, sent in enumerate(X_test):
    # print(''.join(itr['word'] for itr in sent))
    j = 0
    while j < len(sent):
        fos = []
        tmp = []
        flag = 'OFF'
        while j < len(sent):
            item = sent[j]
            token = item['word']
            netag = y_pred[i][j]
            if item['postag'] == 'wp':
                j += 1
                break
            if netag == 'B-T':
                if tmp:
                    s = ''.join(tmp)
                    fos.append((s, flag))
                    tmp = []
                tmp.append(token)
                flag = 'T'
            elif netag == 'B-O':
                if tmp:
                    s = ''.join(tmp)
                    fos.append((s, flag))
                    tmp = []
                tmp.append(token)
                flag = 'O'
            elif netag == 'I-T':
                tmp.append(token)
            elif netag == 'I-O':
                tmp.append(token)
            elif netag == 'OFF':
                if tmp:
                    s = ''.join(tmp)
                    fos.append((s, flag))
                    tmp = []
                flag = 'OFF'
            j += 1
        if tmp:
            s = ''.join(tmp)
            fos.append((s, flag))
            tmp = []
        if not fos:
            continue
        idx = idx_test[i]
        # print(str(idx) + '\t', end='')
        # for itr in fos:
        #     print(itr[1], end='')
        # # print('\t'+''.join(itr['word'] for itr in sent))
        # print('\t'+'\t'.join("%s:%s" % (token, netag) for (token, netag) in fos))

        sta = []
        for itr in fos:
            if not sta:
                sta.append(itr)
            elif itr[1] == 'T':
                if sta[-1][1] == 'O':
                    fo_result.append((idx, itr[0], sta[-1][0]))
                    del sta[-1]
                else:
                    sta.append(itr)
            elif itr[1] == 'O':
                if sta[-1][1] == 'T':
                    fo_result.append((idx, sta[-1][0], itr[0]))
                    del sta[-1]
                else:
                    sta.append(itr)
        for itr in sta:
            if itr[1] == 'T':
                fo_result.append((idx, itr[0], '_'))
            else:
                fo_result.append((idx, '_', itr[0]))

fo_result = sorted(list(set(fo_result)))
# cnt = 0
with open(TASK1_ANS, 'w', encoding='utf-8') as f:
    for itr in fo_result:
        # cnt += 1
        # print("%d,%s,%s" % (itr[0], itr[1], itr[2]))
        f.write("%d,%s,%s\n" % (itr[0], itr[1], itr[2]))
        # if cnt >= 20:
        #     break