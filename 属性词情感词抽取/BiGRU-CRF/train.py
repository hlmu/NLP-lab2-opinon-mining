#%%
import paddle.fluid as fluid
import paddle
import numpy as np

import sys
import os
import math

from paddle.fluid.initializer import NormalInitializer

from utils import str2bool, get_logger, get_entity, to_lodtensor
from data import read_corpus, read_dictionary, tag2label, random_embedding, data_reader, sentence2id, vocab_build
from config import FEATURE_SENTS_TRAIN, FEATURE_SENTS_TEST, CORPUS_TRAIN

import jieba

train_data_path = FEATURE_SENTS_TRAIN
test_data_path = FEATURE_SENTS_TEST

vocab_path = './data/word2id.pkl'
model_path = './model_/'

checkpoint_path = './checkpoint/model.ckpt'
checkpoint_dir ='./checkpoint/'


batch_size = 256
hidden_dim = 256
epochs = 100
embedding_dim = 300
embedding_lr = 1
crf_lr = 0.2
bigru_num = 2
learning_rate = 0.001
clip = 5.0
dropout_rate = 0.2
updata_embedding = True
use_gpu = False

def get_vocab(vocab_path):
    vocab = read_dictionary(vocab_path)
    return vocab

def prepare_data(train_data_path, test_data_path, vocab):
    train_data = read_corpus(train_data_path)
    test_data = read_corpus(test_data_path)

    train_rader = paddle.batch(data_reader(train_data, vocab, tag2label), batch_size)
    test_reader = paddle.batch(data_reader(test_data, vocab, tag2label, shuffle=False), batch_size)

    return train_rader, test_reader

# def bilstm_net(data,
#                label,
#                dict_dim,
#                emb_dim,
#                hid_dim,
#                emb_lr,
#                dropout_rate,
#                num_tags):
#     # 加了dropout层的embedding层
#     emb = fluid.layers.embedding(
#         input=data,
#         size=[dict_dim, emb_dim],
#         param_attr=fluid.ParamAttr(learning_rate=emb_lr))
#     emb = fluid.layers.dropout(emb, dropout_prob=dropout_rate)

#     fc0 = fluid.layers.fc(input=emb, size=hid_dim * 4)
#     rfc0 = fluid.layers.fc(input=emb, size=hid_dim * 4)

#     lstm_h, c = fluid.layers.dynamic_lstm(
#         input=fc0, size=hid_dim * 4, is_reverse=False)
#     rlstm_h, c = fluid.layers.dynamic_lstm(
#         input=rfc0, size=hid_dim * 4, is_reverse=True)
#     # extract last layer
#     lstm_last = fluid.layers.sequence_last_step(input=lstm_h)
#     rlstm_last = fluid.layers.sequence_last_step(input=rlstm_h)

#     lstm_concat = fluid.layers.concat(input=[lstm_last, rlstm_last], axis=-1)
    
#     shape = fluid.layers.shape(lstm_concat)

#     output = fluid.layers.reshape(lstm_concat, [-1, 2 * hid_dim])

#     fc1 = fluid.layers.fc(input=output, size=num_tags)

#     # logits = fluid.layers.reshape(fc1, [-1, shape[0], num_tags])
#     crf_cost = fluid.layers.linear_chain_crf(
#     input=fc1,
#     label=label
#     )
    
#     avg_cost = fluid.layers.mean(x=crf_cost)

#     return avg_cost



def lex_net(emb_dim,
        grnn_hidden_dim,
        emb_learning_rate,
        crf_learning_rate,
        bigru_num,
        word_dict_len,
        label_dict_len):
    """
    define the lexical analysis network structure
    """
    word_emb_dim = emb_dim
    grnn_hidden_dim = grnn_hidden_dim
    emb_lr = emb_learning_rate
    crf_lr = crf_learning_rate
    bigru_num = bigru_num
    init_bound = 0.1
    IS_SPARSE = True

    def _bigru_layer(input_feature):
        """
        define the bidirectional gru layer
        """
        pre_gru = fluid.layers.fc(
            input=input_feature,
            size=grnn_hidden_dim * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        gru = fluid.layers.dynamic_gru(
            input=pre_gru,
            size=grnn_hidden_dim,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        pre_gru_r = fluid.layers.fc(
            input=input_feature,
            size=grnn_hidden_dim * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        gru_r = fluid.layers.dynamic_gru(
            input=pre_gru_r,
            size=grnn_hidden_dim,
            is_reverse=True,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        bi_merge = fluid.layers.concat(input=[gru, gru_r], axis=1)
        return bi_merge

    def _net_conf(word, target):
        """
        Configure the network
        """
        word_embedding = fluid.layers.embedding(
            input=word,
            size=[word_dict_len, word_emb_dim],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr=fluid.ParamAttr(
                learning_rate=emb_lr,
                name="word_emb",
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound)))

        input_feature = word_embedding
        for i in range(bigru_num):
            bigru_output = _bigru_layer(input_feature)
            input_feature = bigru_output

        emission = fluid.layers.fc(
            size=label_dict_len,
            input=bigru_output,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        crf_cost = fluid.layers.linear_chain_crf(
            input=emission,
            label=target,
            param_attr=fluid.ParamAttr(
                name='crfw',
                learning_rate=crf_lr))
        crf_decode = fluid.layers.crf_decoding(
            input=emission, param_attr=fluid.ParamAttr(name='crfw'))
        avg_cost = fluid.layers.mean(x=crf_cost)
        return avg_cost, crf_decode

    word = fluid.layers.data(
        name='word', shape=[1], dtype='int64', lod_level=1)
    target = fluid.layers.data(
        name="target", shape=[1], dtype='int64', lod_level=1)

    avg_cost, crf_decode= _net_conf(word, target)

    return avg_cost, crf_decode, word, target

# def train(train_rader, vocab, use_gpu, epochs, save_dirname=model_path):
#     # word seq data
#     data = fluid.layers.data(
#         name="words", shape=[1], dtype="int64", lod_level=1)
#     # label data
#     label = fluid.layers.data(name="label", shape=[1], dtype="int64", lod_level=1)

#     cost = bilstm_net(data, label, len(vocab), embedding_dim, hidden_dim, embedding_lr, dropout_rate, len(tag2label))

#     # cost,_,_,_ = lex_net(embedding_dim, hidden_dim, embedding_lr, crf_lr, bigru_num, len(vocab), len(tag2label))

#     cost = fluid.layers.mean(cost)

#     # optimizer = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate)
#     # optimizer.minimize(cost)

#     sgd_optimizer = fluid.optimizer.SGD(learning_rate=learning_rate)
#     sgd_optimizer.minimize(cost)

#     place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
#     exe = fluid.Executor(place)
#     exe.run(fluid.default_startup_program())
#     feeder = fluid.DataFeeder(feed_list=[data, label], place=place)

#     for i in range(epochs):
#         data_size = 0
#         total_cost = 0.0
#         num = 0
#         for data in train_rader():
#             num += 1
#             avg_cost_np = exe.run(
#                 feed=feeder.feed(data), 
#                 fetch_list=[cost.name]
#             )
#             data_size = len(data)
#             total_cost +=  np.sum(avg_cost_np) / data_size
#             print('batch:%d, avg_cost:%f' % (num, np.sum(avg_cost_np)/data_size))
#         print("[train info]: epochs: %d, , avg_cost: %f" %
#               (i, total_cost / num))
#         epoch_model = save_dirname + "/" + "epoch" + str(i)
#         fluid.io.save_inference_model(epoch_model, ["words"], exe)

def train_lex(vocab, tag2label, train_rader, epochs, use_gpu, model_path, checkpoint_dir, checkpoint_path):
    cost, crf_decode, word, target = lex_net(embedding_dim, hidden_dim, embedding_lr, crf_lr, bigru_num, len(vocab)+1, len(tag2label)+1)
    cost = fluid.layers.mean(cost)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=learning_rate)
    sgd_optimizer.minimize(cost)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    feeder = fluid.DataFeeder(feed_list=[word, target], place=place)

    prog = fluid.default_main_program()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb+') as f:
            program_desc_str = f.read()
        prog = fluid.Program.parse_from_string(program_desc_str)
        fluid.io.load_persistables(exe, checkpoint_dir, prog)
        print('load checkpoint successfully')
    
    for i in range(epochs):
        data_size = 0
        total_cost = 0.0
        num = 0
        for data in train_rader():
            num += 1
            avg_cost_np = exe.run(
                feed=feeder.feed(data), 
                fetch_list=[cost.name]
            )
            data_size = len(data)
            total_cost +=  np.sum(avg_cost_np) / data_size
            print('batch:%d, avg_cost:%f' % (num, np.sum(avg_cost_np)/data_size))

            # if num == 50: 
            #     # 保存训练快照
            #     with open(checkpoint_path, "wb+") as f:
            #         f.write(prog.desc.serialize_to_string())
        
            #     fluid.io.save_persistables(exe, checkpoint_dir, prog)
            #     print('save checkpoint successfully')
        print("[train info]: epochs: %d, , avg_cost: %f" %
              (i, total_cost / num))
        epoch_model = model_path + "/" + "epoch" + str(i)
        fluid.io.save_inference_model(epoch_model, ["word"], target_vars=crf_decode, executor=exe)

        # 保存训练快照
        with open(checkpoint_path, "wb+") as f:
            f.write(prog.desc.serialize_to_string())
 
        fluid.io.save_persistables(exe, checkpoint_dir, prog)
        print('save checkpoint successfully')


#%%
if __name__ == "__main__":
    vocab_build(vocab_path, CORPUS_TRAIN)
    vocab = get_vocab(vocab_path)
    train_reader, test_reader = prepare_data(train_data_path, test_data_path, vocab)
    # train(train_reader, vocab, False, 30)
    train_lex(vocab, tag2label, train_reader, epochs, use_gpu, model_path, checkpoint_dir, checkpoint_path)