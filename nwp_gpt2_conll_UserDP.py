
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import json
import csv
import logging
logging.getLogger().setLevel(logging.CRITICAL)
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from transformers import AdamW, get_linear_schedule_with_warmup
import time
from copy import deepcopy
import math
import torch.nn as nn
import argparse
import pickle
import random

parser = argparse.ArgumentParser(description='PyTorch GTP-2 AG')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--epochs', type=int, default=1000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                    help='batch size')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
# parser.add_argument('--cuda', default=False,
#                     help='use CUDA')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--ns', type=float, default=2.5,
                    help='gradient clipping')
parser.add_argument('--nu', type=int, default=4,
                    help='number of users per round')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='initial learning rate')
parser.add_argument('--loadpre', type=int, default=0,
                    help='load pretrained WT103 model')
parser.add_argument('--rs', type=int, default=0,
                    help='resume epoch')
args = parser.parse_args()
args.tied = True

resume_epoch = args.rs

BATCH_SIZE = 1000
# EPOCHS = 5
LEARNING_RATE = args.lr
WARMUP_STEPS = 5000 #50
# MAX_SEQ_LEN = 50
model_type = "gpt2"
max_ = 128#354
S = args.clip # 'the clip bound of the gradients'
user_per_epoch = args.nu
noise_scale = args.ns
ent_percent = 0.5 #0.5



path_csv = 'results_nwp/conll/UserDP/conll_' + model_type + '_bs' + str(BATCH_SIZE)  + '_wt' + str(WARMUP_STEPS) + '_lr' + str(LEARNING_RATE) + '_max' + str(max_) + '_clip' + str(S) + \
     '_ns' + str(noise_scale) + '_nu' + str(args.nu)+ '_seed' + str(args.seed) 

print('LEARNING_RATE', LEARNING_RATE)
print('max_', max_)
print('clip',S)
print('path_csv',path_csv)
# print('epochs', epochs)
print('clip', S)
print('nu', user_per_epoch)


list_epoch = [1,48,95,142,189]#4, 2.5 # GPT-2  [0.18,0.185,...,0.2]

nepochs = list_epoch[-1] #37217 # 335


def choose_from_top(probs, n=1):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

class CONLL_data(Dataset):
    def __init__(self, name, model_type, max_length, dataset_path = '../../data/conll_rw2/'):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        self.tokenized_data = []

        data_path = os.path.join(dataset_path, name)
        df = pd.read_csv(data_path)
        text = df['text']

        for row in text:
            self.tokenized_data.append(torch.tensor(self.tokenizer.encode(f"{row[:max_length]}<|endoftext|>"))) 
        
    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, item):
        return self.tokenized_data[item]

def evaluate(eval_data):
    print("\nEvaluating...")
    model.eval()
    sum_loss = 0.0
    tmp_seq = None
    # print(len(data_dataloader))
    # exit()
    with torch.no_grad():
        for i in range(len(eval_data)):
            #################### "Fit as many joke sequences into MAX_SEQ_LEN sequence as possible" logic start ####
            eval_data_seq = eval_data[i].unsqueeze(0).to(device)
            # #The first data sequence in the sequence
            # if not torch.is_tensor(tmp_seq):
            #     tmp_seq = deepcopy(seq)
            #     continue
            # else:
            #     #The next joke does not fit in so we process the sequence and leave the last joke 
            #     #as the start for next sequence 
            #     if tmp_seq.size()[1] + seq.size()[1] > MAX_SEQ_LEN:
            #         eval_data_seq = deepcopy(tmp_seq)
            #         tmp_seq = deepcopy(seq)
            #     else:
            #         #Add the joke to sequence, continue and try to add more
            #         tmp_seq = torch.cat([tmp_seq, seq[:,1:]], dim=1)
            #         continue
            ################## Sequence ready, process it through the model ##################
            # eval_data_seq = eval_data_seq.to(device)
            outputs = model(eval_data_seq, labels=eval_data_seq)
            # outputs=model(eval_data_seq[:-1], labels=eval_data_seq[1:])
            # outputs=model(eval_data_seq)
            loss, logits = outputs[:2] #outputs[:2] = outputs[0]
            sum_loss += loss.detach().data

    return sum_loss/len(eval_data), torch.exp(sum_loss/len(eval_data))


def train(train_data, train_idx, model, S):
    model_local.load_state_dict(model.state_dict(), strict=False)
    # local_before =  deepcopy(model_local.state_dict())
    # print('local_before before train', local_before['fc2.bias'])

    model_local.train()
    sum_loss = 0.0

    for i in train_idx:
        train_data_seq = train_data[i].unsqueeze(0).to(device)
        ################## Sequence ready, process it through the model ##################
        train_data_seq = train_data_seq.to(device)
        outputs = model_local(train_data_seq, labels=train_data_seq)
        # outputs=model(train_data_seq[:-1], labels=train_data_seq[1:])
        # outputs=model(train_data_seq)
        loss, logits = outputs[:2] #outputs[:2] = outputs[0]
        loss.backward()
        sum_loss += loss.detach().data

    torch.nn.utils.clip_grad_norm_(model_local.parameters(), S)
    optimizer_local.step()
    # scheduler.step() 
    optimizer_local.zero_grad()
    model_local.zero_grad()

    loss_train = sum_loss/len(train_idx)

    global_w = deepcopy(model.state_dict())
    local_w =  deepcopy(model_local.state_dict())
    differ_w = deepcopy(model.state_dict())

    for k in differ_w.keys():
        differ_w[k] = local_w[k] - global_w[k]

    return loss_train, differ_w







def FedAvg3(w_b, w_c, m):
    w_avg = deepcopy(w_b)
    for k in w_avg.keys():
        w_avg[k] = w_b[k] + w_c[k] * m
    return w_avg

def get_data_ue_ent_ag_classification_qs( i, user_, ent_active,dict_ent, ent_no_active, qs, evaluation=False):
    # qs = 0.5
    have_sent = []
    for e in ent_active:
        if e.item() in dict_ent:
            # print(i.item())
            # print(type(i.item()))
            tmp = dict_ent[e.item()]
            have_sent.extend(tmp)

    other_sent = []
    for e in ent_no_active:
        if e in dict_ent:
            # print(i.item())
            # print(type(i.item()))
            tmp = dict_ent[e]
            other_sent.extend(tmp)

    samples = user_[i]
    # print('samples',samples)

    in_have_sent = [s for s in samples if s in have_sent] # Sentitive use in training
    not_sent = [s for s in samples if s not in other_sent] # Sentitive but not selected => not use it in training
    not_sent = [s for s in not_sent if s not in have_sent] # Non-sensitive
    # not_sent = []
    n = int(len(not_sent)*qs)
    train_idx = random.sample(list(not_sent), n)
    train_idx.extend(in_have_sent)

    return train_idx




# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)


tmp = np.load('../../data/trainUserDataCount_CONLL.npz',allow_pickle=True)
num_user_train = tmp['user_idx']
tmp = np.load('../../data/testUserDataCount_CONLL.npz',allow_pickle=True)
num_user_test = tmp['user_idx']
# tmp = np.load('../../data/validUserDataCount_AG2.npz',allow_pickle=True)
# num_user_valid = tmp['user_idx']
# flat_list = [item for sublist in num_user_train for item in sublist]
# print(max(flat_list))
# print(min(flat_list))
# print(len(flat_list))
# exit()

num_user = len(num_user_train) # 'number of users N1'
print('num_user', num_user)
# print('73 ', num_user_train[73])
sr_user = user_per_epoch/num_user

# # Min = 1
w_u = np.ones((num_user,), dtype=int) 
Wu = sum(w_u)
# w_u = num_user_train*mul/max(num_user_train)
# Wu = sum(w_u)
qW = sr_user*Wu


print('Wu', Wu)
print('sr_user', sr_user)
print('qW', qW)

sensitivity = S/qW
# sensitivity = sensitivity / qE # sensitivity is \mathbb{S}
batch_size = args.batch_size # 'batch_size L'
std_ = sensitivity*noise_scale
print('S', S)
print('sensitivity', sensitivity)
print('std_', std_)
print('std_^2', std_**2)







device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

start_time = time.time()
train_data = CONLL_data('train_conll.csv', model_type, max_) 
# val_data = AG_data('valid_preprocess.csv', model_type, 1024) 
test_data = CONLL_data('test_conll.csv', model_type, max_) 
# train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=1, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

# print(type(train_data))
# print(train_data[0])
# exit()

print('--- Read data in ',  time.time()-start_time)

# print(len(train_data))
# print(train_data[0])
# print(train_data[1])
# exit()

start_time = time.time()
tokenizer = GPT2Tokenizer.from_pretrained(model_type)
model = GPT2LMHeadModel.from_pretrained(model_type)
model = model.to(device)


model_local = GPT2LMHeadModel.from_pretrained(model_type)
model_local = model_local.to(device)

# if resume_epoch > 0:
#     model.load_state_dict(torch.load(path_csv[:-3] + '.pt'))
#     model_local.load_state_dict(torch.load(path_csv[:-3] + '.pt'))


print('--- Load model in ',  time.time()-start_time)
for param in model.parameters():
    param.requires_grad = True
for param in model_local.parameters():
    param.requires_grad = True


optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
optimizer_local = AdamW(model_local.parameters(), lr=LEARNING_RATE)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps = -1)


# models_folder = "trained_models"
# if not os.path.exists(models_folder):
#     os.mkdir(models_folder)

train_losses=[]
train_ppl=[]
test_losses=[]
test_ppl=[]
all_iters = []
# iter_ = 0

c = 0
for epoch in range( nepochs + 1):

    print('\n Epoch {:} / {:}'.format(epoch + 1, nepochs))

    model.train()
    w_update = model.state_dict()
    for t in w_update.keys():
        w_update[t] = w_update[t].float()

    model.load_state_dict(w_update)

    diff_out_locals = []
    user_active = torch.randperm(num_user)[:user_per_epoch]#user_per_epoch]#[:user_per_epoch] #[:batch_size]

    print('-------------------')
    # print('-------------------')
    # print('epoch ', epoch)
    
    k = 0
    # print('user_active ', len(user_active))
    # print('ent_active ', len(ent_active))
    user_data = []
    user_targets = []
    user_len = []
    # print('user_active',user_active.data)

    epoch_start_time = time.time()
    train_loss = 0
    train_l = float('inf')

    for i in user_active:
        c += 1
        # print('user ',i)
        train_idx = num_user_train[i]

        train_l, diff_out = train(train_data, train_idx, model, S)#loss_train
        if k == 0:
            diff_out_locals = deepcopy(diff_out)
            for t in diff_out_locals.keys():
                diff_out_locals[t] = diff_out_locals[t] * w_u[i]
        else: 
            diff_out_locals = FedAvg3(diff_out_locals, diff_out, w_u[i])
        k += 1
        train_loss += train_l

    print('Done each epoch in ', time.time()-epoch_start_time)


    train_loss = train_loss/len(user_active)
    print(f"avg loss train {train_loss}")
    print(f"ppl {torch.exp(train_loss)}")
    print('---')

    diff_glob = deepcopy(diff_out_locals)
    for t in diff_glob.keys():
        diff_glob[t] = torch.div(diff_glob[t], qW)

    for t in w_update.keys():
        noise = torch.empty(w_update[t].size()).normal_(mean=0.0, std=std_**2).cuda()
        w_update[t] += diff_glob[t] + noise

    model.load_state_dict(w_update)

    if  epoch+1 in list_epoch:# or epoch == nepochs: #epoch % 10 == 0 or 
        
        
        loss_test, ppl_test = evaluate(test_data)  
        print(f"sum loss test {loss_test}")
        print(f"ppl {ppl_test}")
        print('---')

        train_losses.append(train_loss.to('cpu').numpy())
        train_ppl.append(torch.exp(train_loss).to('cpu').numpy())
        all_iters.append(epoch+1)
        test_losses.append(loss_test.to('cpu').numpy())
        test_ppl.append(ppl_test.to('cpu').numpy())

        data_w = {'iters': all_iters, 'test ppl': test_ppl, 'test loss': test_losses, 'train loss': train_losses, 'train ppl': train_ppl}  
        my_csv = pd.DataFrame(data_w)
        my_csv.to_csv(path_csv + '.csv', index=False )
        torch.save(model.state_dict(), os.path.join('', f"" + path_csv + ".pt") )


        
# Store the model after each epoch to compare the performance of them
torch.save(model.state_dict(), os.path.join('', f"" + path_csv + ".pt") )

print('LEARNING_RATE', LEARNING_RATE)
print('max_', max_)
print('clip',S)
print('path_csv',path_csv)
# print('epochs', epochs)
print('clip', S)
print('nu', user_per_epoch)












