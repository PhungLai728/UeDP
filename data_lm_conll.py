import os
import torch
import pandas as pd
from collections import Counter
import re


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = {}
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter.setdefault(word, 0)
        self.counter[word] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, seq_len, bsz, bsz_eval, bsz_test):
        self.dictionary = Dictionary()
        import pickle
        # name = path.split('/')[2]
        if os.path.exists('data/conll_rw2/dictionary_conll'):
            with open('data/conll_rw2/dictionary_conll', 'rb') as file:
                self.dictionary = pickle.load(file)
        else:
            self.tokenize(os.path.join(path, 'train.txt'))
            self.tokenize(os.path.join(path, 'valid.txt'))
            self.tokenize(os.path.join(path, 'test.txt'))
            new_dict = [(self.dictionary.counter[i], i) for i in self.dictionary.word2idx]
            new_dict.sort(key=lambda x: x[0])
            new_dict.reverse()
            for i in range(len(new_dict)):
                self.dictionary.word2idx[new_dict[i][1]] = i
                self.dictionary.idx2word[i] = new_dict[i][1] 
                self.dictionary.counter[i] = new_dict[i][0]
            tkens = len(new_dict)
            # self.dictionary.word2idx['<unk>'] = tkens
            # self.dictionary.idx2word.append('<unk>')
            # self.dictionary.counter[tkens] = new_dict[0][0]
            with open('data/conll_rw2/dictionary_conll', 'wb') as file:
                pickle.dump(self.dictionary, file)

    
        #print(self.dictionary.word2idx)
        self.train_lm,self.train_len   = self.tokenize_(os.path.join(path, 'train.txt'), seq_len, bsz)
        self.valid_lm,self.valid_len  = self.tokenize_(os.path.join(path, 'valid.txt'), seq_len, bsz_eval)
        self.test_lm,self.test_len  = self.tokenize_(os.path.join(path, 'test.txt'), seq_len, bsz_test)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() 
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        return
        
    def tokenize_(self, path, seq_len, bsz):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary

        with open(path, 'r') as f:
            count = 0
            for line in f:
                count += 1
        count = count // bsz * bsz

        with open(path, 'r') as f:
            ids = torch.LongTensor(count*(seq_len+1))
            token = 0
            sent = 0
            len_gt = torch.LongTensor(count)
            for line in f:
                words = line.split() 
                if token < count*(seq_len+1):
                    if len(words) < seq_len and len(words) > 0:
                        len_gt[sent] = len(words)
                    elif len(words) < seq_len and len(words) == 0:
                        len_gt[sent] = 1
                    else:
                        len_gt[sent] = seq_len
                    sent += 1
                    if len(words) > seq_len: # >= seq_len + 1 (i.e., 11)
                        for w in range(seq_len+1):
                            word = words[w]
                            if word in self.dictionary.word2idx:
                                ids[token] =  self.dictionary.word2idx[word]
                            else:
                                ids[token] =  self.dictionary.word2idx['<unk>']
                            # ids[token] =  self.dictionary.word2idx[word]
                            token += 1
                    else:
                        add = seq_len + 1 - len(words)
                        zeros = [0]*add
                        for word in words:
                            if word in self.dictionary.word2idx:
                                ids[token] =  self.dictionary.word2idx[word]
                            else:
                                ids[token] =  self.dictionary.word2idx['<unk>']
                            token += 1
                        for _ in zeros:
                            ids[token] = 0
                            token += 1
                else:
                    break
                  
            

        return ids, len_gt 



