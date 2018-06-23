from sklearn.metrics import confusion_matrix
import numpy as np
import logging

class Evaluator2(object):
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.clear()

    def keys(self):
        return ['binary']

    def clear(self):
        self.cm2 = np.zeros((2, 2), dtype=int)

    def accumulate(self, solution, pred):
        def label_2to2(probs):
            assert len(probs.shape) == 2
            assert probs.shape[1] == 2
            preds = np.argmax(probs, axis=1)
            return preds

        def get_cm2(solution, pred):
            solution = label_2to2(solution)
            pred = label_2to2(pred)
            self.cm2 += confusion_matrix(solution, pred, [0, 1])
            return solution == pred

        return {'binary': int(get_cm2(solution, pred))}

    def evaluate(self, solution, pred):
        clear()
        accumulate(solution, pred)

    def statistic(self):
        cm2 = self.cm2

        binary_total = float(np.sum(cm2))

        ret = {}
        ret['binary'] = (cm2[0, 0] + cm2[1, 1]) / binary_total
        if self.verbose:
            logging.info('Cm2:\n%s' % self.cm2)

        return ret

class Evaluator5(object):
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.clear()

    def clear(self):
        self.cm5 = np.zeros((5, 5), dtype=int)
        self.cm2 = np.zeros((5, 5), dtype=int)

    def keys(self):
        return ['binary', 'fine-grained']

    def accumulate(self, solution, pred):
        def label_5to5(probs):
            assert len(probs.shape) == 2
            assert probs.shape[1] == 5
            preds = np.argmax(probs, axis=1)
            return preds

        def label_5to2(probs):
            assert len(probs.shape) == 2
            assert probs.shape[1] == 5
            probs_without2 = probs - np.array([0.0, 0.0, 0.99999, 0.0, 0.0])
            preds = np.argmax(probs_without2, axis=1)
            return preds

        def get_cm5(solution, pred):
            solution = label_5to5(solution)
            pred = label_5to5(pred)
            self.cm5 += confusion_matrix(solution, pred, [0, 1, 2, 3, 4])
            return solution == pred

        def get_cm2(solution, pred):
            solution = label_5to2(solution)
            pred = label_5to2(pred)
            self.cm2 += confusion_matrix(solution, pred, [0, 1, 2, 3, 4])
            return solution == pred

        return {'binary': int(get_cm2(solution, pred)), \
                'fine-grained': int(get_cm5(solution, pred))}

    def evaluate(self, solution, pred):
        clear()
        accumulate(solution, pred)

    def statistic(self):
        cm5 = self.cm5
        cm2 = self.cm2

        fine_grained_total = float(np.sum(cm5))
        binary_total = float(np.sum(cm2) - np.sum(cm2[2, 0:5]))

        ret = {}
        ret['fine-grained'] = np.sum([cm5[i][i] for i in range(5)]) / fine_grained_total
        ret['binary'] = (np.sum(cm2[0:2, 0:2]) + np.sum(cm2[3:5, 3:5])) / binary_total
        if self.verbose:
            logging.info('Cm5:\n%s' % self.cm5)
            logging.info('Cm2:\n%s' % self.cm2)

        return ret

EvaluatorList = {2: Evaluator2, 5: Evaluator5}

