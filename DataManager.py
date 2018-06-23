import numpy as np
import theano
import codecs
import random
random.seed(1229)

class DataManager(object):
    def __init__(self, data, words):
        def load_data(fname):
            data = []
            with open(fname) as f:
                for line in f:
                    now = line.strip().split()
                    data.append((int(now[0]), now[1:]))
            return data

        self.origin_data = {}
        for fname in ['train', 'dev', 'test']:
            self.origin_data[fname] = load_data('dataset/%s/%s.txt' % (data, fname))
        self.origin_words = {}
        for key, fname in words.items():
            self.origin_words[key] = load_data('dataset/wordlist/%s' % fname)
    
    def gen_word_list(self):
        self.words = {}
        for key in ['negation', 'intensifier', 'sentiment']:
            words = {}
            for label, text in self.origin_words[key]:
                if repr(text) not in words.keys():
                    words[repr(text)] = int(label)
            self.words[key] = words
        words = {}
        for key in ['train', 'dev', 'test']:
            for label, sent in self.origin_data[key]:
                for word in sent:
                    if repr([word]) not in words.keys():
                        words[repr([word])] = 0
        self.words['words'] = words
        return self.words

    def gen_data(self):
        self.real_words = {'negation':{}, 'intensifier':{}, 'sentiment':{}, 'words':{}}
        def match(sent, l=3):
            for case, key in enumerate(['negation', 'intensifier', 'sentiment', 'words']):
                for length in reversed(range(1, l+1)):
                    now = repr(sent[:length])
                    if now in self.words[key].keys():
                        if now in self.real_words[key].keys():
                            subcase, index = self.real_words[key][now]
                            return [case, subcase, index], length
                        else:
                            subcase = self.words[key][now]
                            index = len(self.real_words[key])
                            self.real_words[key][now] = (subcase, index)
                            return [case, subcase, index], length
        self.grained = 1 + max([x for data in self.origin_data.values() for x, y in data])

        self.data = {}
        for key in ['train', 'dev', 'test']:
            data = []
            label = []
            for rating, sent in self.origin_data[key]:
                result = []
                while len(sent) > 0:
                    res, length = match(sent)
                    result.append(res)
                    sent = sent[length:]
                data.append(np.array(result))
                rat = np.zeros((self.grained), dtype=theano.config.floatX)
                rat[rating] = 1
                label.append(rat)
            self.data[key] = (label, data)
        
        self.data['train_small'] = self.data['train'][0][::10], self.data['train'][1][::10], 
        
        self.words = self.real_words
        self.index = list(range(len(self.data['train'][0])))
        self.index_now = 0
        return self.data

    def get_mini_batch(self, mini_batch_size=25):
        if self.index_now >= len(self.index):
            random.shuffle(self.index)
            self.index_now = 0
        st, ed = self.index_now, self.index_now + mini_batch_size
        label = np.take(self.data['train'][0], self.index[st:ed], 0)
        data = np.take(self.data['train'][1], self.index[st:ed], 0)
        self.index_now += mini_batch_size
        return label, data

    def analysis(self):
        num_sent = np.array([0, 0, 0, 0])
        num_word = np.array([0, 0, 0, 0])
        n_sent = 0
        for label, data in self.data.values():
            n_sent += len(data)
            for sent in data:
                now = np.bincount(sent[:, 0], minlength=4)
                num_sent += np.min([now, np.ones_like(now)], 0)
                num_word += now
        return num_sent[:3] / n_sent, num_word[:3] / n_sent

    def gen_analysis_data(self, fname):
        def match(sent, l=3):
            for case, key in enumerate(['negation', 'intensifier', 'sentiment', 'words']):
                for length in reversed(range(1, l+1)):
                    now = repr(sent[:length])
                    if now in self.words[key].keys():
                        if now in self.real_words[key].keys():
                            subcase, index = self.real_words[key][now]
                            return [case, subcase, index], length
                        else:
                            print('error')
        with codecs.open(fname, "r", encoding='utf-8', errors='ignore') as fdata:
            origin_data = fdata.readlines()
            data = []
            for line in origin_data:
                sent = line.strip().split()
                result = []
                while len(sent) > 0:
                    res, length = match(sent)
                    result.append(res)
                    sent = sent[length:]
                data.append(np.array(result))
        return origin_data, data

if __name__ == '__main__':
    data = DataManager('sst', {'negation': 'negation.txt', \
            'intensifier': 'intensifier.txt', \
            'sentiment': 'sentiment.txt'})
    data.gen_word_list()
    data.gen_data()
    negation = [[] for i in range(32)]
    intensity = [[] for i in range(30)]
    def form(s):
        t = []
        for i in s:
            if (i == '.') or (i == ','):
                break
            else:
                t.append(i)
        return t
    for key in ['train', 'dev', 'test']:
        sents = data.data[key][1]
        origins = data.origin_data[key]
        for sent, origin in zip(sents, origins):
            for i, line in enumerate(sent.tolist()):
                if line[0] == 0:
                    word = line[2]
                    negation[word].append(origin[1][i:])
                if line[0] == 1:
                    word = line[2]
                    intensity[word].append(origin[1][i:])
    #for key, value in data.words['negation'].items():
    #    name = '_'.join(eval(key))
    #    print(name)
    #    with open('dataset/analysis/negation/%s.txt' % name, 'w') as f:
    #        for sent in negation[value[1]]:
    #            s = form(sent)
    #            if len(s) > 1:
    #                f.writelines(' '.join(s)+'\n')
    for key, value in data.words['intensifier'].items():
        name = '_'.join(eval(key))
        with open('dataset/analysis/intensifier/%s.txt' % name, 'w') as f:
            for sent in intensity[value[1]]:
                s = form(sent)
                if len(s) > 1:
                    f.writelines(' '.join(s)+'\n')
            
