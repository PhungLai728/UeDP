import os
import torch
import pandas as pd
from collections import Counter


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
        name = path.split('/')[2]
        if os.path.exists('dictionary_' + name + '_ag_30k'):
            with open('dictionary_' + name + '_ag_30k', 'rb') as file:
                self.dictionary = pickle.load(file)
        else:
            self.tokenize(os.path.join(path, 'train_preprocess.csv'))
            self.tokenize(os.path.join(path, 'valid_preprocess.csv'))
            self.tokenize(os.path.join(path, 'test_preprocess.csv'))

            new_dict = [(self.dictionary.counter[i], i) for i in self.dictionary.word2idx]
            new_dict.sort(key=lambda x: x[0])
            new_dict.reverse()
            tkens = 30000
            new_dict = [new_dict[m] for m in range(tkens-1)]

            self.dictionary.word2idx = {}
            self.dictionary.idx2word = []
            self.dictionary.counter = {}

            for i in range(len(new_dict)):
                self.dictionary.word2idx[new_dict[i][1]] = i
                self.dictionary.idx2word.append(new_dict[i][1])
                self.dictionary.counter[i] = new_dict[i][0]
            self.dictionary.word2idx['<unk>'] = tkens-1
            self.dictionary.idx2word.append('<unk>')
            self.dictionary.counter[tkens-1] = new_dict[0][0]

            with open('dictionary_' + name + '_ag_30k', 'wb') as file:
                pickle.dump(self.dictionary, file)
        #print(self.dictionary.word2idx)
        self.train_lm,self.train_len,self.train,self.train_label  = self.tokenize_(os.path.join(path, 'train_preprocess.csv'), seq_len, bsz)
        self.valid_lm,self.valid_len,self.valid,self.valid_label = self.tokenize_(os.path.join(path, 'valid_preprocess.csv'), seq_len, bsz_eval)
        self.test_lm,self.test_len,self.test,self.test_label = self.tokenize_(os.path.join(path, 'test_preprocess.csv'), seq_len, bsz_test)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        all_info = pd.read_csv(path)
        all_news = all_info['Description']
        for i in range(len(all_news)):
            # print(i)
            words = all_news[i].split()
            for word in words:
                self.dictionary.add_word(word)
        return
        
    def tokenize_(self, path, seq_len, bsz):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        all_info = pd.read_csv(path)
        all_news = all_info['Description']
        all_labels = all_info['Class Index']
        all_labels = [i for i in all_labels]

        count = len(all_news)
        count = count // bsz * bsz

        # Tokenize file content
        ids = torch.LongTensor(count*(seq_len+1))
        ids_seq = torch.LongTensor(count*seq_len)
        token = 0
        token_seq = 0
        sent = 0
        len_gt = torch.LongTensor(count)
        label = torch.LongTensor(count)
        for i in range(len(all_news)):
            line = all_news[i]
            words = line.split() 
            tmp = []
            if token < count*(seq_len+1):
                label[sent] = all_labels[i]
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
                        token += 1
                        if w != seq_len:
                            if word in self.dictionary.word2idx:
                                ids_seq[token_seq] =  self.dictionary.word2idx[word]
                            else:
                                ids_seq[token_seq] =  self.dictionary.word2idx['<unk>']
                            token_seq += 1
                    
                else:
                    add = seq_len + 1 - len(words)
                    zeros = [0]*add
                    zeros_seq = [0]*(add-1)
                    for word in words:
                    
                        if word in self.dictionary.word2idx:
                            ids[token] =  self.dictionary.word2idx[word]
                        else:
                            ids[token] =  self.dictionary.word2idx['<unk>']

                        token += 1
                        if word in self.dictionary.word2idx:
                            ids_seq[token_seq] =  self.dictionary.word2idx[word]
                        else:
                            ids_seq[token_seq] =  self.dictionary.word2idx['<unk>']
                        token_seq += 1
                    for _ in zeros:
                        ids[token] = 0
                        token += 1
                    for _ in zeros_seq:
                        ids_seq[token_seq] = 0
                        token_seq += 1
            else:
                break
                  
            

        return ids, len_gt, ids_seq, label



