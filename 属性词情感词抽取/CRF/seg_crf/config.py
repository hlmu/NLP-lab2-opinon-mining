"""
配置文件，记录项目中各个文件的路径
"""
# ltp模型目录的路径
LTP_DATA_DIR = './model'  
# 原始数据路径
TRAIN_LABEL = 'data/Train_labels.csv'
TRAIN_REVIEW = 'data/Train_reviews.csv'
TEST_REVIEW = 'data/Test_reviews.csv'
# TEST_LABEL = 'results/Test_labels.csv'
# 问题一输出
TASK1_ANS = 'results/task1_answer.csv'
# 观点词典路径
OPINION_DIC = 'data/opinion_dic.csv'
# 属性词典路径
FEATURE_DIC = 'data/feature_dic.csv'
# 句子分词特征文件
FEATURE_SENTS_TRAIN = 'data/feature_sents_train.pkl'
FEATURE_SENTS_TEST = 'data/feature_sents_test.pkl'
# 句子特征观点标记模型
FOR_MODEL = 'data/fo_recognition.pkl'