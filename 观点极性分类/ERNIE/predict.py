#coding:utf-8
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning on classification task """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import numpy as np
import os
import time
import paddle
import paddle.fluid as fluid
import paddlehub as hub
from dataset import MyData
from config import SENTIMENTS
from config import TASK2_ANS
from config import TASK3_ANS
import codecs
import csv

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--batch_size",     type=int,   default=1, help="Total examples' number in batch for training.")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=False, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--use_pyreader", type=ast.literal_eval, default=False, help="Whether use pyreader to feed data.")
parser.add_argument("--dataset", type=str, default="chnsenticorp", help="The choice of dataset")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=False, help="Whether use data parallel.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    dataset = None
    metrics_choices = []
    # Download dataset and use ClassifyReader to read dataset
    if args.dataset.lower() == "fo_senti":
        dataset = MyData()
        module = hub.Module(name="ernie")
        metrics_choices = ["acc"]
    elif args.dataset.lower() == "chnsenticorp":
        dataset = hub.dataset.ChnSentiCorp()
        module = hub.Module(name="ernie_tiny")
        metrics_choices = ["acc"]
    elif args.dataset.lower() == "tnews":
        dataset = hub.dataset.TNews()
        module = hub.Module(name="roberta_wwm_ext_chinese_L-24_H-1024_A-16")
        metrics_choices = ["acc"]
    elif args.dataset.lower() == 'inews':
        dataset = hub.dataset.INews()
        module = hub.Module(name="roberta_wwm_ext_chinese_L-24_H-1024_A-16")
        metrics_choices = ["acc"]
    elif args.dataset.lower() == 'thucnews':
        dataset = hub.dataset.THUCNEWS()
        module = hub.Module(name="roberta_wwm_ext_chinese_L-24_H-1024_A-16")
        metrics_choices = ["acc"]
    elif args.dataset.lower() == 'iflytek':
        dataset = hub.dataset.IFLYTEK()
        module = hub.Module(name="roberta_wwm_ext_chinese_L-24_H-1024_A-16")
        metrics_choices = ["acc"]
    # The first metric will be choose to eval. Ref: task.py:799
    elif args.dataset.lower().startswith("xnli"):
        dataset = hub.dataset.XNLI(language=args.dataset.lower()[-2:])
        module = hub.Module(name="roberta_wwm_ext_chinese_L-24_H-1024_A-16")
        metrics_choices = ["acc"]
    else:
        raise ValueError("%s dataset is not defined" % args.dataset)

    support_metrics = ["acc", "f1", "matthews"]
    for metric in metrics_choices:
        if metric not in support_metrics:
            raise ValueError("\"%s\" metric is not defined" % metric)

    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)
    reader = hub.reader.ClassifyReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len)

    # Construct transfer learning network
    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_output" for token-level output.
    pooled_output = outputs["pooled_output"]

    # Setup feed list for data feeder
    # Must feed all the tensor of ERNIE's module need
    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name,
    ]

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        use_data_parallel=False,
        use_pyreader=args.use_pyreader,
        use_cuda=args.use_gpu,
        batch_size=args.batch_size,
        enable_memory_optim=False,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.finetune.strategy.DefaultFinetuneStrategy())

    # Define a classfication finetune task by PaddleHub's API
    cls_task = hub.TextClassifierTask(
        data_reader=reader,
        feature=pooled_output,
        feed_list=feed_list,
        num_classes=dataset.num_labels,
        config=config,
        metrics_choices=metrics_choices)

    # Data to be prdicted
    # data = [[d.text_a, d.text_b] for d in dataset.get_dev_examples()[:3]]
    
    data = []
    task2_ans = []
    with codecs.open(TASK2_ANS, "r", encoding="UTF-8") as f:
        reader = csv.reader(f)
        examples = []
        seq_id = 0
        for line in reader:
            feature = line[1] if line[1] != '_' else ''
            opinion = line[2] if line[2] != '_' else ''
            task2_ans.append((line[0], line[1], line[2], line[3]))
            data.append([feature+opinion])


    with open(TASK3_ANS, 'w', encoding='utf-8') as f:
        index = 0
        run_states = cls_task.predict(data=data)
        results = [run_state.run_results for run_state in run_states]
        for batch_result in results:
            # get predict index
            batch_result = np.argmax(batch_result, axis=2)[0]
            for result in batch_result:
                # print("%s\tpredict=%s" % (data[index][0], result))
                # f.write("%s,%s\n" % (data[index][0], CATEGORIES[result]))
                f.write("%s,%s,%s,%s,%s\n" % (task2_ans[index][0], task2_ans[index][1], task2_ans[index][2], task2_ans[index][3], SENTIMENTS[result]))
                index += 1
