import numpy as np
import logging

class ADADELTA(object):
    def __init__(self, params, lr=1, lr_word_vector=0.1, lr_decay=0.95, epsilon=1e-6):
        logging.info('Optimizer ADADELTA lr %f lr_decay %f epsilon %f' % (lr, lr_decay, epsilon))
        self.lr = lr
        self.lr_word_vector = lr_word_vector
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.acc_grad = {}
        self.acc_update = {}
        for param in params:
            self.acc_grad[param] = np.zeros_like(param.get_value())
            self.acc_update[param] = np.zeros_like(param.get_value())

    def iterate(self, grads):
        lr = self.lr
        lr_decay = self.lr_decay
        epsilon = self.epsilon
        for param, grad in grads.items():
            if param.name[0] == 'V':
                param.set_value(param.get_value() - grad.get_value() * self.lr_word_vector)
            else:
                self.acc_grad[param] = lr_decay * self.acc_grad[param] + \
                        (1 - lr_decay) * (grad.get_value())**2
                param_update = np.sqrt(self.acc_update[param] + epsilon) \
                        / np.sqrt(self.acc_grad[param] + epsilon) * grad.get_value() * lr
                self.acc_update[param] = lr_decay * self.acc_update[param] \
                        + (1 - lr_decay) * param_update**2
                param.set_value(param.get_value() - param_update)

class ADAGRAD(object):
    def __init__(self, params, lr, lr_word_vector=0.1, epsilon=1e-10):
        logging.info('Optimizer ADAGRAD lr %f' % (lr, ))
        self.lr = lr
        self.lr_word_vector = lr_word_vector
        self.epsilon = epsilon
        self.acc_grad = {}
        for param in params:
            self.acc_grad[param] = np.zeros_like(param.get_value())

    def iterate(self, grads):
        lr = self.lr
        epsilon = self.epsilon
        for param, grad in grads.items():
            if param.name[0] == 'V':
                param.set_value(param.get_value() - grad.get_value() * self.lr_word_vector)
            else:
                self.acc_grad[param] = self.acc_grad[param] + grad.get_value()**2
                param_update = lr * grad.get_value() / (np.sqrt(self.acc_grad[param]) + epsilon)
                param.set_value(param.get_value() - param_update)

class SGD(object):
    def __init__(self, params, lr, lr_word_vector=0.1, momentum=0.9):
        logging.info('Optimizer SGD lr %s momentum %s' % (lr, momentum))
        self.lr = lr
        self.lr_word_vector = lr_word_vector
        self.momentum = momentum
        self.sum_grad = {}
        for param in params:
            self.sum_grad[param] = np.zeros_like(param.get_value())

    def iterate(self, grads):
        lr = self.lr
        momentum = self.momentum
        for param, grad in grads.items():
            if param.name[0] == 'V':
                param.set_value(param.get_value() - grad.get_value() * self.lr_word_vector)
            else:
                self.sum_grad[param] = self.sum_grad[param] * momentum + lr * grad.get_value()
                param.set_value(param.get_value() - self.sum_grad[param])
                grad.set_value(np.zeros_like(param.get_value()))

OptimizerList = {'SGD': SGD, 'ADAGRAD': ADAGRAD, 'ADADELTA': ADADELTA}
