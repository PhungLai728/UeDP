
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
parser.add_argument('--nu', type=int, default=3,
                    help='number of users per round')
parser.add_argument('--ent', type=str, default='loc',
                    help='type of entity')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='initial learning rate')
parser.add_argument('--loadpre', type=int, default=0,
                    help='load pretrained WT103 model')
parser.add_argument('--rs', type=int, default=0,
                    help='resume epoch')
parser.add_argument('--qs', type=float, default=1.0,
                    help='qs')
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
ent_type =  args.ent
ent_percent = 0.5 #0.5
qs = args.qs



path_csv = 'results_nwp/conll/UeDP/conll_' + model_type + '_bs' + str(BATCH_SIZE)  + '_wt' + str(WARMUP_STEPS) + '_lr' + str(LEARNING_RATE) + '_max' + str(max_) + '_clip' + str(S) + \
    '_e' + ent_type + '_ep' + str(ent_percent) + '_ns' + str(noise_scale) + '_nu' + str(args.nu)+ '_seed' + str(args.seed) + '_qs' + str(args.qs)  

print('LEARNING_RATE', LEARNING_RATE)
print('max_', max_)
print('clip',S)
print('path_csv',path_csv)
# print('epochs', epochs)
print('clip', S)
print('nu', user_per_epoch)
print('ent', ent_type)
print('qs', qs)


if ent_type == 'person':
    # list_epoch = [236,383, 537, 711, 907,1125]#,4513]#,1151,1154,1649,1650,1651,1654] # 1000
    # [0.5,0.6,0.7,0.8,0.9,1,2]
    # list_epoch = [13,180,372,563,755] #600 # Bert [0.3,0.35,0.4,0.45,0.5]
    # list_epoch = [1,20,40]#,59,79,99,118,138,158,177,197,223]#25, 2.5 # GPT-2   [0.18,0.182,...,0.19]
    list_epoch = [1,50]#,99,148,197]#25, 2.5 # GPT-2  [0.18,0.185,...,0.2]

elif ent_type == 'org':
    # list_epoch = [73,137,201,270,347,434]#,1786] #500 [0.5,0.6,0.7,0.8,0.9,1,2]
    # list_epoch = [5,160,331,503,677] #200 # Bert [0.3,0.35,0.4,0.45,0.5] 
    # list_epoch = [1,23,45]#,67,90,112,134,156,178,201]# #20, 2.5 # GPT-2   [0.18,0.182,...,0.19]
    list_epoch = [1,56]#,112,167,223]#20, 2.5 # GPT-2  [0.18,0.185,...,0.2]

elif ent_type == 'loc':
    # list_epoch = [195,324,456,603,771,958]#,3854] #500 [0.5,0.6,0.7,0.8,0.9,1,2]
    # list_epoch = [31,253,484,716,947] #250 # Bert [0.3,0.35,0.4,0.45,0.5]
    # list_epoch = [1,21,41]#,61,81,102,122,142,162,182,202] #20, 2.5 # GPT-2   [0.18,0.182,...,0.19]
    list_epoch = [1,51]#,102,152,202]#20, 2.5 # GPT-2  [0.18,0.185,...,0.2]

# elif ent_type == 'all':
#     list_epoch = [51,100,150,203,262,329]#,1367] #500 [0.5,0.6,0.7,0.8,0.9,1,2]
elif ent_type == 'misc':
    # list_epoch = [167,279,396,525,672,836]#,3373] #500
    # list_epoch = [18,206,405,614,823] #250 # Bert [0.3,0.35,0.4,0.45,0.5]
    # list_epoch = [1,21,42]#,63,84,105,126,146,167,188,209]#30, 2.5 # GPT-2  [0.18,0.182,...,0.19]
    list_epoch = [1,53]#,105,157,209]#30, 2.5 # GPT-2  [0.18,0.185,...,0.2]

elif ent_type == 'all_':
    # list_epoch = [49,97,145,254,319,1327] #500  
    # list_epoch = [26,248,476,703,930] #150 # Bert [0.3,0.35,0.4,0.45,0.5] 
    # list_epoch = [4,156,322,488,663] #175 # Bert [0.3,0.35,0.4,0.45,0.5] 
    # list_epoch = [1,20,39]#,58,77,96,115,134,153,172,191] #10, 2.5 # GPT-2  [0.18,0.182,...,0.19]
    list_epoch = [1,48]#,96,143,191]#10, 2.5 # GPT-2  [0.18,0.185,...,0.2]

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


all_sent_dicts =pickle.load(open("../../data/conll_rw2/conll_sensitive_dicts", "rb"))  
# ent_type = 'all'
if ent_type == 'loc':
    dict_ent = all_sent_dicts[2]
elif ent_type == 'person':
    dict_ent = all_sent_dicts[1]
elif ent_type == 'misc':
    dict_ent = all_sent_dicts[3]
elif ent_type == 'org':
    dict_ent = all_sent_dicts[0]
else:
    dict_0 = all_sent_dicts[0]
    dict_1 = all_sent_dicts[1]
    dict_2 = all_sent_dicts[2]
    dict_3 = all_sent_dicts[3]

    dict_ent = {}
    for d in [dict_0, dict_1, dict_2, dict_3]:
      dict_ent.update(d)

num_user = len(num_user_train) # 'number of users N1'
print('num_user', num_user)
# print('73 ', num_user_train[73])
sr_user = user_per_epoch/num_user


if ent_type == 'org':
    num_entity = 5187
elif ent_type == 'misc':
    num_entity = 3567
elif ent_type == 'person':
    num_entity = 4406
elif ent_type == 'loc':
    num_entity = 5433
elif ent_type == 'all_':
    num_entity = 11176

#     non_ent_list = [a for a in non_ent if a not in  loc_list and a not in gpe_list and a not in product_list ]
        # and a not in org_list and a not in date_list and a not in person_list and a not in fac_list]

sr_entity = ent_percent #ent_per_epoch/num_entity # e.g., 0.5
ent_per_epoch = int(ent_percent* len(dict_ent))

# tmp = np.load('non_ent.npz',allow_pickle=True)
# non_ent_list = tmp['non_ent_list']
num_noEntity = 14040 - num_entity# 'number of entities N2'
# num_noEntity = 78431
sr_non_entity = qs
print('sample ent rate ', sr_entity)
print('num_entity ', num_entity)
print('ent_per_epoch', ent_per_epoch)
print('num_noEntity ', num_noEntity)
# # Min = 1
w_u = np.ones((num_user,), dtype=int) 
Wu = sum(w_u)
# w_u = num_user_train*mul/max(num_user_train)
# Wu = sum(w_u)
qW = sr_user*Wu

# w_e = num_ent_train*mul/max(num_ent_train)
# We = sum(w_e)
# w_ne = num_noEnt_train*mul/max(num_noEnt_train)
# Wne = sum(w_ne)
w_e = np.ones((num_entity,), dtype=int) 
We = sum(w_e)

w_ne = np.ones((num_noEntity,), dtype=int) 
Wne = sum(w_ne)

qE = sr_entity*We + sr_non_entity*Wne

print('Wu', Wu)
print('sr_user', sr_user)
print('qW', qW)
print('We', We)
print('Wne', Wne)
print('sr_entity', sr_entity)
print('sr_non_entity', sr_non_entity)
print('qE', qE)

sensitivity = (user_per_epoch+1)* max(w_u)*S/ (qW *qE)
# sensitivity = sensitivity / qE # sensitivity is \mathbb{S}
batch_size = args.batch_size # 'batch_size L'
std_ = sensitivity*noise_scale
print('ent type', ent_type)
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
    ent_active = torch.randperm(len(dict_ent))[:ent_per_epoch]
    ent_no_active = [mm for mm in range(len(dict_ent)) if mm not in ent_active]


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
        
        train_idx = get_data_ue_ent_ag_classification_qs(i, num_user_train, ent_active,dict_ent, ent_no_active, qs, evaluation=False)
        #train model
        # train_loss, _ = train(S, train_idx, model)
        # print(len(train_idx))
        # print((train_idx))
        # print(k)
        # exit()


        if k == 0 and len(train_idx) > 0:
            train_l, diff_out = train(train_data, train_idx, model, S)
            diff_out_locals = deepcopy(diff_out)
            for t in diff_out_locals.keys():
                diff_out_locals[t] = diff_out_locals[t] * w_u[i]
        elif k == 0 and len(train_idx) == 0:
            diff_out_locals = model.state_dict()
            for t in diff_out_locals.keys():
                diff_out_locals[t] = diff_out_locals[t] * 0.0
        elif k > 0 and len(train_idx) == 0:
            diff_out = model.state_dict()
            for t in diff_out.keys():
                diff_out[t] = diff_out[t] * 0.0
            diff_out_locals = FedAvg3(diff_out_locals, diff_out, 0.0)
        else: # k > 0 and len(train_sample_idx) > 0:
            train_l, diff_out = train(train_data, train_idx, model, S)
            diff_out_locals = FedAvg3(diff_out_locals, diff_out, w_u[i])
        # print('train_l',train_l)
        # print('--')

        train_loss += train_l
        k += 1

    print('Done each epoch in ', time.time()-epoch_start_time)


    train_loss = train_loss/len(user_active)
    print(f"avg loss train {train_loss}")
    print(f"ppl {torch.exp(train_loss)}")
    print('---')

    diff_glob = deepcopy(diff_out_locals)
    for t in diff_glob.keys():
        diff_glob[t] = torch.div(diff_glob[t], (qW*qE))

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
print('ent', ent_type)
print('qs', qs)













