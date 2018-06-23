import theano
import theano.tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import argparse
import logging
import time
import collections
from WordLoader import WordLoader

class RLstm(object):
    def __init__(self, words, grained, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', type=str, default='test')
        parser.add_argument('--rseed', type=int, default=int(1000*time.time()) % 19921229)
        parser.add_argument('--dim_hidden', type=int, default=300)
        parser.add_argument('--dim_leaf', type=int, default=300)
        parser.add_argument('--dropout', type=int, default=1)
        parser.add_argument('--regular', type=float, default=0.0001)
        parser.add_argument('--word_vector', type=str, default='wordvector/glove.refine.txt')
        parser.add_argument('--innear', type=float, default=5)

        args, _ = parser.parse_known_args(argv)

        self.name = args.name
        logging.info('Model init: %s' % self.name)
        
        self.srng = RandomStreams(seed=args.rseed)
        logging.info('RandomStream seed %d' % args.rseed)
        
        self.dim_leaf = args.dim_leaf
        self.dim_hidden = args.dim_hidden
        logging.info('dim: hidden=%s, leaf=%s' % (self.dim_hidden, self.dim_leaf))

        self.grained = grained
        logging.info('grained: %s' % self.grained)

        self.dropout = args.dropout
        logging.info('dropout: %s' % self.dropout)

        self.regular = args.regular
        logging.info('l2 regular: %s' % self.regular)
        
        self.margin = -np.log(1.0 / self.grained) * 2
        logging.info('margin: %s' % self.margin)
        
        self.innear = args.innear
        logging.info('innear: %s' % self.innear)

        self.words = words
        self.num = {key: len(value) for key, value in words.items()}
        logging.info('vocabulary size: %s' % self.num)

        self.init_param()
        self.load_word_vector('dataset/' + args.word_vector)
        self.init_function()
    
    def init_param(self):
        def shared_matrix(dim, name, u=0, b=0):
            matrix = self.srng.uniform(dim, low=-u, high=u, dtype=theano.config.floatX) + b
            f = theano.function([], matrix)
            return theano.shared(f(), name=name)

        u = lambda x : 1 / np.sqrt(x)

        dh, dl = self.dim_hidden, self.dim_leaf
        v_num, self.V = [], []
        for key in ['negation', 'intensifier', 'sentiment', 'words']:
            v_num.append(self.num[key])
            self.V.append(shared_matrix((self.num[key], dl), 'V'+key, 0.01))
        v_num = [sum(v_num[:i])for i in range(len(self.num))]
        self.v_num = shared(np.array(v_num))
        self.V_all = T.concatenate(self.V, 0)

        self.W_i=shared_matrix((dl, dh),'W_i',u(dl))
        self.U_i=shared_matrix((dh, dh),'U_i',u(dh))
        self.b_i=shared_matrix((dh, ),'b_i',0.)

        self.W_f=shared_matrix((dl, dh),'W_f',u(dl))
        self.U_f=shared_matrix((dh, dh),'U_f',u(dh))
        self.b_f=shared_matrix((dh, ),'b_f',0.,1.)

        self.W_c=shared_matrix((dl, dh),'W_c',u(dl))
        self.U_c=shared_matrix((dh, dh),'U_c',u(dh))
        self.b_c=shared_matrix((dh, ),'b_c',0.)

        self.W_o=shared_matrix((dl, dh),'W_o',u(dl))
        self.U_o=shared_matrix((dh, dh),'U_o',u(dh))
        self.b_o=shared_matrix((dh, ),'b_o',0.)

        self.W_hy=shared_matrix((dh, self.grained),'W_hy',u(dh))
        self.b_hy=shared_matrix((self.grained, ),'b_hy',0.)
        
        self.params = self.V + [self.W_i, self.U_i, self.b_i, \
                self.W_f, self.U_f, self.b_f, \
                self.W_c, self.U_c, self.b_c, \
                self.W_o, self.U_o, self.b_o, \
                self.W_hy, self.b_hy]
        
        #add W_neg and W_int into params
        self.W_neg=shared_matrix((self.num['negation'],self.grained,self.grained),'W_neg', .0)
        self.W_int=shared_matrix((self.num['intensifier'],self.grained,self.grained),'W_int', .0)
        self.params.append(self.W_neg)
        self.params.append(self.W_int)
        
        #add sentiment_vector into params
        self.sentiment_vector=shared_matrix((5, self.grained), 'v_sent', .0)
        self.params.append(self.sentiment_vector)

        if self.grained == 2:
            sentiment_vector = np.array([\
                    [0.5, -0.5], \
                    [0.25, -0.25], \
                    [.0, .0], \
                    [-0.25, 0.25], \
                    [-0.5, 0.5]], \
                    dtype=theano.config.floatX)
            neg_value = np.array([[-1, 1], [1, -1]])
            W_neg = np.array([neg_value]*self.num['negation'], dtype=theano.config.floatX)
            int_value = np.array([[2, -1], [-1, 2]])
            W_int = np.array([int_value]*self.num['intensifier'], dtype=theano.config.floatX)
        elif self.grained == 5:
            sentiment_vector = np.eye(5, dtype=theano.config.floatX) - 0.2
            neg_value = np.array([[0, 0, 0, 0, 1], \
                    [0, 0, 0, 1, 0], \
                    [0, 0, 1, 0, 0], \
                    [0, 1, 0, 0, 0], \
                    [1, 0, 0, 0, 0]])
            W_neg = np.array([neg_value]*self.num['negation'], dtype=theano.config.floatX)
            int_value = np.array([[1, 1, 0, 0, 0], \
                    [0, 0, 0, 0, 0], \
                    [0, 0, 1, 0, 0], \
                    [0, 0, 0, 0, 0], \
                    [0, 0, 0, 1, 1]])
            W_int = np.array([int_value]*self.num['intensifier'], dtype=theano.config.floatX)
        else:
            logging.info('grained is not 2 or 5')
        self.sentiment_vector.set_value(sentiment_vector)
        self.W_neg.set_value(W_neg)
        self.W_int.set_value(W_int)

    def load_word_vector(self, fname):
        logging.info('loading word vectors...')
        loader = WordLoader()
        dic = loader.load_word_vector(fname)

        for v, key in zip(self.V, ['negation', 'intensifier', 'sentiment', 'words']):
            value = v.get_value()
            not_found = 0
        
            for words, index in self.words[key].items():
                word_list = eval(words)
                if (len(word_list) == 1) and (word_list[0] in dic.keys()):
                    value[index[1]] = list(dic[word_list[0]])
                else:
                    not_found += 1
            
            logging.info('word vector for %s, %d not found.' % (key, not_found))
            v.set_value(value)

    def init_function(self):
        logging.info('init function...')
        
        self.data = T.lmatrix()
        self.label = T.vector()
        
        def encode(t, h_prev, c_prev):
            x_t = self.V_all[self.v_num[t[0]] + t[2]]
            i_t = T.nnet.sigmoid(T.dot(x_t, self.W_i) + T.dot(h_prev, self.U_i) + self.b_i)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.W_f) + T.dot(h_prev, self.U_f) + self.b_f)
            c_c = T.tanh(T.dot(x_t, self.W_c) + T.dot(h_prev, self.U_c) + self.b_c)
            c = i_t * c_c + f_t * c_prev
            o_t = T.nnet.sigmoid(T.dot(x_t, self.W_o) + T.dot(h_prev, self.U_o) + self.b_o)
            h = o_t * T.tanh(c)
            return h, c

        
        [hf_history, _], _ = theano.scan(encode, sequences=self.data, \
                outputs_info=[dict(initial=T.zeros(self.dim_hidden)),\
                dict(initial=T.zeros(self.dim_hidden))])
        
        pred = T.nnet.softmax(T.dot(hf_history, self.W_hy) + self.b_hy)
        unit = theano.shared(np.array([[1.0/self.grained]*self.grained], \
                dtype=theano.config.floatX))
        pred_last = T.concatenate([unit, pred[:-1]], 0)

        def loss_kl(last, now, t):
            index = T.switch(T.eq(T.transpose(self.data)[0], t), 1, 0)
            dis = - T.sum(last * T.log(now), 1) - T.sum(T.log(last) * now, 1)
            return T.mean(T.maximum(0, dis - self.margin) * index) * self.innear

        self.loss_common = loss_kl(pred_last, pred, 3)
        
        v_sent = T.take(self.sentiment_vector, self.data[:, 0], 0)
        pred_last_sentiment = T.nnet.softmax(pred_last + v_sent)
        self.loss_sentiment = loss_kl(pred_last_sentiment, pred, 2)

        w_neg = T.take(self.W_neg, T.minimum(self.num['negation'] - 1, self.data[:, 2]), 0)
        pred_last_negation = T.nnet.softmax(T.batched_tensordot(w_neg, pred_last, [[1], [1]]))
        self.loss_negation = loss_kl(pred_last_negation, pred, 0)
        
        w_int = T.take(self.W_int, T.minimum(self.num['intensifier'] - 1, self.data[:, 2]), 0)
        pred_last_intensifier = T.nnet.softmax(T.batched_tensordot(w_int, pred_last, [[1], [1]]))
        self.loss_intensifier = loss_kl(pred_last_intensifier, pred, 1)

        self.loss_innear = self.loss_common + self.loss_sentiment \
                + self.loss_negation + self.loss_intensifier

        hf=hf_history[-1]
        embedding=hf

        self.use_noise = theano.shared(np.asarray(0., dtype=theano.config.floatX))

        if self.dropout == 1:
            embedding_for_train = embedding * self.srng.binomial(embedding.shape, \
                    p = 0.5, n = 1, dtype=embedding.dtype)
            embedding_for_test = embedding * 0.5
        else:
            embedding_for_train = embedding
            embedding_for_test = embedding

        self.pred_for_train = T.nnet.softmax(T.dot(embedding_for_train, self.W_hy) + self.b_hy)[0]
        self.pred_for_test = T.nnet.softmax(T.dot(embedding_for_test, self.W_hy) + self.b_hy)[0]

        self.l2 = sum([T.sum(param**2) for param in self.params]) \
                - sum([T.sum(param**2) for param in self.V])
        self.loss_supervised = -T.sum(self.label * T.log(self.pred_for_train))
        self.loss_l2 = 0.5 * self.l2 * self.regular
        self.loss = self.loss_supervised + self.loss_l2 + self.loss_innear

        logging.info('getting grads...')
        grads = T.grad(self.loss, self.params)
        self.updates = collections.OrderedDict()
        self.grad = {}
        for param, grad in zip(self.params, grads):
            g = theano.shared(np.asarray(np.zeros_like(param.get_value()), \
                    dtype=theano.config.floatX))
            self.grad[param] = g
            self.updates[g] = g + grad

        logging.info("compiling func of train...")
        self.func_train = theano.function(
                inputs = [self.label, self.data],
                outputs = [self.loss, self.loss_supervised, self.loss_l2, self.loss_innear, \
                        self.loss_common, self.loss_sentiment, \
                        self.loss_negation, self.loss_intensifier],
                updates = self.updates,
                on_unused_input='warn')

        logging.info("compiling func of test...")
        self.func_test = theano.function(
                inputs = [self.label, self.data],
                outputs = [self.loss_supervised, self.pred_for_test],
                on_unused_input='warn')
    def dump(self, epoch):
        import scipy.io
        mdict = {}
        for param in self.params:
            val = param.get_value()
            mdict[param.name] = val
        scipy.io.savemat('mat/%s.%s' % (self.name, epoch), mdict=mdict)
    def load(self, fname):
        import scipy.io
        mdict = scipy.io.loadmat('mat/%s.mat' % fname)
        for param in self.params:
            if len(param.get_value().shape) == 1:
                pass
                #param.set_value(np.asarray(mdict[param.name][0], dtype=theano.config.floatX))
            if len(param.get_value().shape) >= 2:
                param.set_value(np.asarray(mdict[param.name], dtype=theano.config.floatX))
