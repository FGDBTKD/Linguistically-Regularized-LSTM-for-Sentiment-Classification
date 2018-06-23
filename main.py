import numpy as np
import theano
import os
import argparse
import logging
import sys
import json
import random
from Optimizer import OptimizerList
from Evaluator import EvaluatorList
from RLstm import RLstm as Model
from DataManager import DataManager

argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='test')
parser.add_argument('--dataset', type=str, default='mr/mr0')
parser.add_argument('--screen', type=int, choices=[0, 1], default=1)
parser.add_argument('--optimizer', type=str, default='ADAGRAD', \
        choices=['SGD', 'ADAGRAD', 'ADADELTA'])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_vector', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=25)
parser.add_argument('--batch_num', type=int, default=2000)
parser.add_argument('--interval', type=int, default=40)
parser.add_argument('--analysis', type=int, default=0)

args, _ = parser.parse_known_args(argv)

logging.basicConfig(
        filename = ('log/%s.log' % args.name) * (1-args.screen),
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s',
        datefmt='%H:%M:%S')
logging.info('command: python ' + ' '.join(sys.argv))

dm = DataManager(args.dataset, {'negation': 'negation.txt', \
            'intensifier': 'intensifier.txt', \
            'sentiment': 'sentiment.txt'})
dm.gen_word_list()
dm.gen_data()
info_sent, info_words = dm.analysis()
logging.info('sentence level for negation/intensifier/sentiment: %s' % info_sent)
logging.info('words level for negation/intensifier/sentiment: %s' % info_words)

model = Model(dm.words, dm.grained, argv)
optimizer = OptimizerList[args.optimizer](model.params, args.lr, args.lr_vector)
Evaluator = EvaluatorList[dm.grained]

def do_test(label, data):
    evaluator = Evaluator()
    loss = .0
    evaluator_neg = Evaluator()
    evaluator_int = Evaluator()

    for item_label, item_data in zip(label, data):
        item_loss, item_pred = model.func_test(item_label, item_data)
        loss += item_loss
        evaluator.accumulate(item_label.reshape((1, -1)),item_pred.reshape(1, -1))
        words = np.bincount(item_data[:, 0])
        if words[0] > 0:
            evaluator_neg.accumulate(item_label.reshape((1, -1)),item_pred.reshape(1, -1))
        if words[1] > 0:
            evaluator_int.accumulate(item_label.reshape((1, -1)),item_pred.reshape(1, -1))
    logging.info('loss: %.4f' % (loss/len(data)))
    
    format_acc = lambda acc: ' '.join(['%s:%.4f' % (key, value) for key, value in acc.items()])
    acc = evaluator.statistic()
    logging.info('acc: %s' % format_acc(acc))
    acc_neg = evaluator_neg.statistic()
    logging.info('acc for negation: %s' % format_acc(acc_neg))
    acc_int = evaluator_int.statistic()
    logging.info('acc for intensification: %s' % format_acc(acc_int))

    return loss/len(data), acc

def do_train(label, data):
    for _, grad in model.grad.items():
        grad.set_value(np.asarray(np.zeros_like(grad.get_value()), \
                dtype=theano.config.floatX))
    loss = []
    for item_label, item_data in zip(label, data):
        item_loss = model.func_train(item_label, item_data)
        loss.append(item_loss)
    for _, grad in model.grad.items():
        grad.set_value(grad.get_value() / len(data))
    optimizer.iterate(model.grad)
    return np.sum(np.array(loss), 0) / len(loss)

details = {'loss_train':[], 'loss_dev':[], 'loss_test':[], \
        'acc_train':[], 'acc_dev':[], 'acc_test':[]}

for i in range(args.batch_num):
    if i % args.interval == 0:
        now = {}
        now['loss_train'], now['acc_train'] = do_test(*dm.data['train_small'])
        now['loss_dev'], now['acc_dev'] = do_test(*dm.data['dev'])
        now['loss_test'], now['acc_test'] = do_test(*dm.data['test'])
        for key, value in now.items():
            details[key].append(value)
        with open('result/%s.txt' % args.name, 'w') as f:
            f.writelines(json.dumps(details))
        model.dump(i / args.interval)

    mini_batch = dm.get_mini_batch(args.batch_size)
    loss = do_train(*mini_batch)
    format_loss = ' '.join(['%.3f' % x for x in loss.tolist()])
    logging.info('loss for batch %d: %s' % (i, format_loss))

