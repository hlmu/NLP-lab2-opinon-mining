"""
配置文件，记录项目中各个文件的路径
"""
# ltp模型目录的路径
LTP_DATA_DIR = 'D:/NLP/lab2/model'  
# 原始数据路径
TRAIN_LABEL = 'data/Train_labels.csv'
TRAIN_REVIEW = 'data/Train_reviews.csv'
TEST_REVIEW = 'data/Test_reviews.csv'
# TEST_LABEL = 'results/Test_labels.csv'
# 问题二输出
TASK2_ANS = 'results/task2_answer.csv'
# 观点词典路径
OPINION_DIC = 'data/opinion_dic.csv'
# 属性词典路径
FEATURE_DIC = 'data/feature_dic.csv'
# 句子分词特征文件
FEATURE_SENTS_TRAIN = 'data/feature_sents_train'
FEATURE_SENTS_TEST = 'data/feature_sents_test'
# 句子特征观点标记模型
FOR_MODEL = 'data/fo_recognition'
# 新增词典
LEXICON = 'data/lexicon'

# 语料库文件
CORPUS_TRAIN = 'data/corpus'
TEST_TEMP_OUTPUT = 'results/temp_out.txt'

# 清洗后的文件
TRAIN_REVIEW_CLEANED = 'data/Train_reviews_cleaned.csv'
TRAIN_LABEL_CLEANED = 'data/Train_labels_cleaned.csv'
TEST_REVIEW_CLEANED = 'data/Test_reviews_cleaned.csv'

TRAIN_DATASET = 'data/train_dataset.csv'
TEST_DATASET = 'data/test_dataset.csv'
DEV_DATASET = 'data/dev_dataset.csv'

TAG2LABEL = {
    '包装':0,
    '成分':1,
    '尺寸':2,
    '服务':3,
    '功效':4,
    '价格':5,
    '气味':6,
    '使用体验':7,
    '物流':8,
    '新鲜度':9,
    '真伪':10,
    '整体':11,
    '其他':12
}

# print(TAG2LABEL.keys())
CATEGORIES = ['包装', '成分', '尺寸', '服务', '功效', '价格', '气味', '使用体验', '物流', '新鲜度', '真伪', '整体', '其他']

TASK1_ANS = 'results/task1_answer.csv'