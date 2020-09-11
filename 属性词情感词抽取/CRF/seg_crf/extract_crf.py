"""
训练CRF属性词特征词标注模型
生成文件位于data/fo_recognition.pkl
"""
from itertools import chain

import sklearn
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import pickle
import scipy
from config import FEATURE_SENTS_TRAIN, FOR_MODEL
from utils.feature_extraction import sent2features, sent2labels

with open(FEATURE_SENTS_TRAIN, 'rb') as f:
    train_sents = pickle.load(f)

X = [sent2features(s) for s in train_sents]
y = [sent2labels(s) for s in train_sents]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, shuffle=False)
crf.fit(X_train, y_train)

# y_pred = crf.predict(X_test)

# for i in range(20):
#     print('\t'.join(item['word'] for item in X_test[i]))
#     print('\t'.join(item for item in y_test[i]))

# for i in range(20):
#     for j in range(len(y_test[i])):
#         if y_test[i][j] != 'OFF':
#             print(X_test[i][j]['word'] + '\t', end='')
#             print(y_test[i][j] + '\t', end='')
#     print()


with open(FOR_MODEL, 'wb') as f:
    pickle.dump(crf, f)

labels = list(crf.classes_) 
labels.remove('OFF')
y_pred = crf.predict(X_test)
print(metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels))

# # define fixed parameters and parameters to search
# crf = sklearn_crfsuite.CRF(
#     algorithm='lbfgs',
#     max_iterations=100,
#     all_possible_transitions=True
# )
# params_space = {
#     'c1': scipy.stats.expon(scale=0.1),
#     'c2': scipy.stats.expon(scale=0.1),
# }

# # use the same metric for evaluation
# f1_scorer = make_scorer(metrics.flat_f1_score,
#                         average='weighted', labels=labels)

# # search
# rs = RandomizedSearchCV(crf, params_space,
#                         cv=3,
#                         verbose=1,
#                         n_jobs=-1,
#                         n_iter=50,
#                         scoring=f1_scorer)
# rs.fit(X_train, y_train)

# print('best params:', rs.best_params_)
# print('best CV score:', rs.best_score_)
# print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))