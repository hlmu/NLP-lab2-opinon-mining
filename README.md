# 实验二：电商评论观点挖掘


## 实验基本情况
### 小组成员
1173710132 牟虹霖
1173710217 侯鹏钰

### 组内分数分配
牟虹霖 1
侯鹏钰 0.95

### 编码要求
本次实验采用UTF-8编码

### 运行所需库(根目录下含requirements.txt)

paddlehub==1.3.0
paddlepaddle==1.5.1
pkuseg==0.0.22
pyltp==0.2.1
python-crfsuite==0.9.6
sklearn==0.0
sklearn-crfsuite==0.3.6
synonyms==3.10.2


## 代码文件说明
### 实验文件结构
```
C:.
├─属性分类
│  └─ERNIE
│      ├─data
│      └─results
├─属性词情感词抽取
│  ├─BiGRU-CRF
│  │  ├─data
│  │  ├─model_
│  │  │  └─epoch99
│  │  └─results
│  ├─CRF
│  │  ├─seg_crf
│  │  │  ├─data
│  │  │  ├─results
│  │  │  └─utils
│  │  └─word_crf
│  │      ├─data
│  │      ├─results
│  │      └─utils
│  └─ERNIE
│      ├─ERNIE_etract
│      │  └─data
│      └─ERNIE_mapping
│          ├─data
│          │  └─.ipynb_checkpoints
│          └─results
└─观点极性分类
    └─ERNIE
        ├─data
        └─results
```

### 项目运行方法
#### 属性词情感词抽取任务——CRF方法

（注：基于字、词的CRF方法运行方法均相同）

运行`python extract_crf.py`指令训练CRF模型

运行`python fo_mapping.py`可于  results  文件夹下生成指令生成  task1_answer.csv  

#### 属性词情感词抽取任务——BiGRU-CRF方法

运行`python gen_feature_sents.py`指令可于  data  文件夹下生成训练所需的数据

运行`python train.py`指令可运行训练模型代码（注：该文件夹已保存一份模型可直接使用）

运行`python test.py`指令可于  results  文件夹下生成  temp_out.txt  为测试中间结果

运行`python fo_mapping.py`指令可于  results  文件夹下生成  task1_answer.csv  文件

#### 属性词情感词抽取任务——ERNIE方法

通过命令行终端进入  属性词情感词抽取/ERNIE/ERNIE_extract  文件夹

运行`sh run_sequence_label.sh`指令即运行模型训练任务

运行`sh run_predict.sh`指令即运行模型的预测任务，可于 results  文件夹下生成temp_out.txt文件

将  temp_out.txt  移动至  属性词情感词抽取/ERNIE/ERNIE_mapping/data  文件夹


通过命令行终端进入  属性词情感词抽取/ERNIE/ERNIE_mapping  文件夹

运行`sh run_classifer.sh`指令即运行模型训练任务

运行`sh run_predict.sh`指令即运行模型的预测任务，可于 results  文件夹下生成  task1_answer.csv  文件

#### 属性分类任务——ERNIE方法

运行`sh run_classifer.sh`指令即运行模型训练任务

运行`sh run_predict.sh`指令即运行模型的预测任务，可于 results  文件夹下生成  task2_answer.csv  文件

#### 观点极性分类任务——ERNIE方法

运行`sh run_classifer.sh`指令即运行模型训练任务

运行`sh run_predict.sh`指令即运行模型的预测任务，可于 results  文件夹下生成  task3_answer.csv  文件

