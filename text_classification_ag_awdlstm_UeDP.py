import torch
print(torch.__version__)
import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import data_lm_ag_glove
import model_classification_glove_freeze
from torch.autograd import Variable
# import torchvision
from torch.utils.tensorboard import SummaryWriter
from utils_ag_ue import batchify_lm, get_batch_lm, repackage_hidden, get_data_user,get_data_ue_ent_ag_classification_qs,get_data_user_classification
import pandas as pd
import copy
import pickle
from weight_drop import WeightDrop
import codecs
import random
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=100,
                    help='size of word embeddings')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--bptt', type=int, default=50,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='report interval')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--log-file', type=str,  default='logs',
                    help='path to save the log')
parser.add_argument('--mmd_kernel_alpha', type=float,  default=0.5,
                    help='mmd kernel')
parser.add_argument('--mmd_lambda', type=float,  default=0.2,
                    help='mmd kernel')
parser.add_argument('--moment', action='store_true',
                    help='using moment regularization')
parser.add_argument('--moment_split', type=int, default=1000,
                    help='threshold for rare and popular words')
parser.add_argument('--moment_lambda', type=float, default=0.02,
                    help='lambda')
parser.add_argument('--adv', action='store_false',
                    help='using adversarial regularization')
parser.add_argument('--adv_bias', type=float, default=1000,
                    help='threshold for rare and popular words')
parser.add_argument('--adv_lambda', type=float, default=0.02,
                    help='lambda')
parser.add_argument('--adv_lr', type=float,  default=0.02,
                    help='adv learning rate')
parser.add_argument('--adv_wdecay', type=float,  default=1.2e-6,
                    help='adv weight decay')
parser.add_argument('--epochs', type=int, default=1000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                    help='batch size')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
# parser.add_argument('--cuda', default=False,
#                     help='use CUDA')
parser.add_argument('--clip', type=float, default=0.1,
                    help='gradient clipping')
parser.add_argument('--nu', type=int, default=500,
                    help='number of users per round')
parser.add_argument('--ent', type=str, default='loc',
                    help='type of entity')
parser.add_argument('--lr', type=float, default=100000,
                    help='initial learning rate')
parser.add_argument('--loadpre', type=int, default=0,
                    help='load pretrained WT103 model')
parser.add_argument('--rs', type=int, default=10,
                    help='resume epoch')
parser.add_argument('--qs', type=float, default=0.5,
                    help='qs')
args = parser.parse_args()
args.tied = True

resume_epoch = args.rs
user_per_epoch = args.nu
clip_bound =  args.clip # 'the clip bound of the gradients'
S = clip_bound #10 #
noise_scale = 2
ent_type =  args.ent

ent_percent = 0.5 #0.5
seq_len = 30

if ent_type == 'gpe':
    list_epoch = [236,383, 537, 711, 907,1125,4513]#,1151,1154,1649,1650,1651,1654] # 1000
    # [0.5,0.6,0.7,0.8,0.9,1,2]
elif ent_type == 'org':
    list_epoch = [73,137,201,270,347,434,1786] #500 [0.5,0.6,0.7,0.8,0.9,1,2]
elif ent_type == 'loc':
    list_epoch = [195,324,456,603,771,958,3854] #500 [0.5,0.6,0.7,0.8,0.9,1,2]
elif ent_type == 'all':
    list_epoch = [51,100,150,203,262,329,1367] #500 [0.5,0.6,0.7,0.8,0.9,1,2]
elif ent_type == 'pii':
    list_epoch = [167,279,396,525,672,836,3373] #500
elif ent_type == 'allpii':
    list_epoch = [49,97,145,254,319,1327] #500  
# else:
#     list_epoch = [45,300,701,1253,1951,2791]
nepochs = list_epoch[-1] #37217 # 335
load_wt103 = args.loadpre


print('user_per_epoch:', user_per_epoch)
print('clip_bound:', clip_bound)
print('ent_type:', ent_type)
print('load_wt103:', load_wt103) # print('args.lr',args.lr)
name = 'ag_lr' + str(args.lr) + '_u' + str(user_per_epoch) + '_e' + ent_type + '_ep' + str(ent_percent) + '_clip' + str(S) + '_ns' + str(noise_scale) + '_emb' + str(args.emsize) + '_len' + str(seq_len) + '_' + str(load_wt103) + 'Wt103_30k_ueDP_glove_gauss_seed' + str(args.seed) + '_qs' + str(args.qs)
# name = name_resume + '_resume' 
dir_path = '../results_classification/AG_UeDP/'
result_path = dir_path + name + '.txt'
writer = SummaryWriter(dir_path + name)
args.save = dir_path + name + '.pt'
# resume = dir_path + name_resume + '.pt'

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, model_local, criterion,  criterion2, optimizer, optimizer2], f)
def model_load(fn):
    global model, model_local, criterion,  criterion2, optimizer, optimizer2
    with open(fn, 'rb') as f:
        model, model_local, criterion,  criterion2, optimizer, optimizer2 = torch.load(f)
        
import os
import hashlib
print('Producing dataset...')
eval_batch_size = 100
test_batch_size = 100
data4ue = np.load('../data/AG/train_ag_ue_30k_seq' + str(seq_len) + '.npz',allow_pickle=True)
train_data = data4ue['ids_seq']
train_len = data4ue['len_gt']
train_target = data4ue['label']
data2 = np.load('../data/AG/test_ag_ue_30k_seq' + str(seq_len) + '.npz',allow_pickle=True)
test_data = data2['ids_seq']
test_len = data2['len_gt']
test_target = data2['label']
data3 = np.load('../data/AG/valid_ag_ue_30k_seq' + str(seq_len) + '.npz',allow_pickle=True)
val_data = data3['ids_seq']
val_len = data3['len_gt']
val_target = data3['label']

###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
criterion = None
with open('../data/AG/dictionary_AG_dataset_ag_30k', 'rb') as file:
    dictionary = pickle.load(file)

word2idx = dictionary.word2idx
embedding_path = '../data/glove.6B.100d.txt'

all_word_embeds = {}
for i, line in enumerate(codecs.open(embedding_path, 'r', 'utf-8')):
    s = line.strip().split()
#     print('s', s)
    if len(s) == args.emsize + 1:
        all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word2idx), args.emsize))
for w in word2idx:
    if w in all_word_embeds:
#         print('w', w)
        word_embeds[word2idx[w]] = all_word_embeds[w]
    elif w.lower() in all_word_embeds:
#         print('w.lower()', w.lower())
        word_embeds[word2idx[w]] = all_word_embeds[w.lower()]
print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

ntokens = len(dictionary)
print('-----------------')
print('-----------------')
print('ntokens ', ntokens)
model = model_classification_glove_freeze.RNNModel(args.model, ntokens, word_embeds, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
model_local = model_classification_glove_freeze.RNNModel(args.model, ntokens, word_embeds, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
                       
if load_wt103 == 1:
    print('start loading wt103')
    lm_wgts_wiki = torch.load('../data/fwd_wt103.h5', map_location=lambda storage, 
                            loc: storage)
    print('start copying')
    new_dict = copy.deepcopy(model.state_dict())
    for k, v in lm_wgts_wiki.items():
        if k == '0.rnns.0.module.weight_hh_l0_raw':
            new_dict[k[2:]] = v
        elif k == '0.rnns.0.module.bias_hh_l0':
            new_dict[k[2:]] = v
        elif k == '0.rnns.1.module.weight_ih_l0':
            new_dict[k[2:]] = v
        elif k == '0.rnns.1.module.bias_ih_l0':
            new_dict[k[2:]] = v
        elif k == '0.rnns.1.module.bias_hh_l0':
            new_dict[k[2:]] = v
        elif k == '0.rnns.1.module.weight_hh_l0_raw':
            new_dict[k[2:]] = v
    model.load_state_dict(new_dict)
###
if args.resume:
    print('Resuming model ...')
    model_load(resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop

criterion = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()
##
if args.cuda:
    print('-------cuda---------')
    model = model.cuda()
    criterion = criterion.cuda()
    model_local = model_local.cuda()
    criterion2 = criterion2.cuda()
##
params = list(model.parameters()) 
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
params2 = list(model_local.parameters()) + list(criterion2.parameters())
total_params2 = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params2 if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

###############################################################################
# Training code
###############################################################################

def FedAvg3(w_b, w_c, m):
    w_avg = copy.deepcopy(w_b)
    for k in w_avg.keys():
        w_avg[k] = w_b[k] + w_c[k] * m
    return w_avg

def removeElements(A, B): 
    n = len(A) 
    return any(A == B[i:i + n] for i in range(len(B)-n + 1)) 

def evaluate(data_source,data_target,  seq_len, user_, len_):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    batch_size = 10
    total_loss = 0
    total_acc = 0

    loss_val = 0
    for u in range(len(user_)):
        data, targets, len_sm = get_data_user_classification(data_source, data_target, len_, u, user_, args, seq_len, evaluation=True)
        loss_tmp = 0
        if data.size(1) < args.batch_size + 1:
            no_iter = 1
        elif data.size(1)%args.batch_size==0:
            no_iter = data.size(1)//args.batch_size
        else:
            no_iter = data.size(1)//args.batch_size + 1

        hidden = model.init_hidden(batch_size)
        acc_u = 0
        for i in range(no_iter):
            if i == no_iter-1:
                start_ = i*batch_size
                end_ = data.size(1)
            else:
                start_ = i*batch_size
                end_ = (i+1)*batch_size

            output, hidden = model(data.t()[start_:end_].t(), hidden, len_sm[start_:end_])
            if len(output.size()) > 2: output = output.view(-1, output.size(2))
            runtime_loss = criterion(output, targets[start_:end_]).data
            num_corrects = (torch.max(output, 1)[1].view(targets[start_:end_].size()).data == targets[start_:end_].data).float().sum()
            acc = num_corrects/targets[start_:end_].size(0)
            acc_u += acc
            hidden = repackage_hidden(hidden)
            loss_tmp += runtime_loss
        total_acc +=   acc_u/no_iter 
        loss_val += loss_tmp/no_iter
    return loss_val.item()/len(user_), total_acc.item()/len(user_)


def train( data, targets, len_bs, model, result_path, epoch, user):
    model_local.load_state_dict(model.state_dict(), strict=False)
    model_local.train()
    inner_product = 0
    count = 0
    save_hiddens = []
    # Turn on training mode which enables dropout.
    start_time = time.time()
    hidden = model_local.init_hidden(args.batch_size)

    if data.size(1) < args.batch_size + 1:
        no_iter = 1
    elif data.size(1)%args.batch_size==0:
        no_iter = data.size(1)//args.batch_size
    else:
        no_iter = data.size(1)//args.batch_size + 1

    loss_train = 0
    for i in range(no_iter):
        loss_ = 0
        no_repeat = 1    
        for _ in range(no_repeat):
            hidden = repackage_hidden(hidden)
            optimizer2.zero_grad()
            if i == no_iter-1:
                start_ = i*args.batch_size
                end_ = data.size(1)
            else:
                start_ = i*args.batch_size
                end_ = (i+1)*args.batch_size

            output, hidden, rnn_hs, dropped_rnn_hs = model_local(data.t()[start_:end_].t(), hidden, len_bs[start_:end_], return_h=True)
            if len(output.size()) > 2: output = output.view(-1, output.size(2))
            loss = criterion2(output, targets[start_:end_])
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            for p in model_local.parameters():
                torch.nn.utils.clip_grad_norm_(p, clip_bound)
            optimizer2.step()
            loss_ += loss.data
        loss_train += loss_/no_repeat

    global_w = copy.deepcopy(model.state_dict())
    local_w =  copy.deepcopy(model_local.state_dict())
    differ_w = copy.deepcopy(model.state_dict())

    for k in differ_w.keys():
        if '_raw' in k:
            differ_w[k] = local_w[k[:-4]] - global_w[k]
        else:
            differ_w[k] = local_w[k] - global_w[k]
    return loss_train, differ_w  

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000
finetune = False
# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    optimizer2 = None
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
        optimizer2 = torch.optim.SGD(params2, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
        optimizer2 = torch.optim.Adam(params2, lr=args.lr, weight_decay=args.wdecay)

    ###############################################################################
    # User-Entity Level DP
    ############################################################################### 
    tmp = np.load('../data/AG/trainUserDataCount_AG2.npz',allow_pickle=True)
    num_user_train = tmp['user_idx']
    tmp = np.load('../data/AG/testUserDataCount_AG2.npz',allow_pickle=True)
    num_user_test = tmp['user_idx']
    tmp = np.load('../data/AG/validUserDataCount_AG2.npz',allow_pickle=True)
    num_user_valid = tmp['user_idx']
    all_sent_dicts =pickle.load(open("../data/AG/ag_sensitive_dicts", "rb"))  
    if ent_type == 'loc':
        dict_ent = all_sent_dicts[2]
    elif ent_type == 'person':
        dict_ent = all_sent_dicts[1]
    elif ent_type == 'gpe':
        dict_ent = all_sent_dicts[3]
    elif ent_type == 'org':
        dict_ent = all_sent_dicts[0]
    else:
        dict_0 = all_sent_dicts[0]
        dict_1 = all_sent_dicts[1]
        dict_2 = all_sent_dicts[2]
        dict_3 = all_sent_dicts[3]

        dict_ent = {}
        for d in [dict_0,dict_1,dict_2, dict_3]:
          dict_ent.update(d)

    num_user = len(num_user_train) # 'number of users N1'
    print('num_user', num_user)
    sr_user = user_per_epoch/num_user
    tmp = np.load('../data/AG/ag_' + ent_type + '_sens_indicator.npz',allow_pickle=True)
    sens_indicator = tmp['sens_indicator']

    if ent_type == 'org':
        num_entity = 58177
        We = data4ue['count_all_org']
    elif ent_type == 'gpe':
        num_entity = 18506
        We = data4ue['count_all_gpe']
    elif ent_type == 'person':
        num_entity = 3639
        We = data4ue['count_all_person']
    elif ent_type == 'loc':
        num_entity = 39988
        We = data4ue['count_all_loc']
    elif ent_type == 'all':
        num_entity = 66195  
        We1 = data4ue['count_all_gpe']
        We2 = data4ue['count_all_org']
        We3 = data4ue['count_all_loc']
        We = We1 +We2+We3
    elif ent_type == 'pii':
        num_entity = 42683 
        We2 = data4ue['count_all_person']
        We3 = data4ue['count_all_loc']
        We = We2+We3
    elif ent_type == 'allpii':
        num_entity =  67157 
        We1 = data4ue['count_all_gpe']
        We2 = data4ue['count_all_person']
        We3 = data4ue['count_all_loc']
        We4 = data4ue['count_all_org']
        We = We1+We2+We3+We4

    sr_entity = ent_percent #ent_per_epoch/num_entity # e.g., 0.5
    ent_per_epoch = int(ent_percent* len(dict_ent))
    num_noEntity = 112000 - num_entity# 'number of entities N2'
    qs = args.qs
    sr_non_entity = qs
    print('sample ent rate ', sr_entity)
    print('num_entity ', num_entity)
    print('ent_per_epoch', ent_per_epoch)
    print('num_noEntity ', num_noEntity)
    # # Min = 1
    w_u = np.ones((num_user,), dtype=int) 
    Wu = sum(w_u)
    qW = sr_user*Wu
    w_e = np.ones((num_entity,), dtype=int) 
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
    
    sensitivity = (num_user+1)* max(w_u)*S/ (qW *qE)
    batch_size = args.batch_size # 'batch_size L'
    std_ = sensitivity*noise_scale
    print('ent type', ent_type)
    print('S', S)
    print('sensitivity', sensitivity)
    print('std_', std_)
    print('std_^2', std_**2)

    iter_ = []
    acc_test_all = []
    loss_test_all = []

    c = 0
    for epoch in range(nepochs + 1):# range(1, args.2epochs+1):
        model.train()
        model_local.train()
        w_update = model.state_dict()

        diff_out_locals = []
        user_active = torch.randperm(num_user)[:user_per_epoch]#user_per_epoch]#[:user_per_epoch] #[:batch_size]
        ent_active = torch.randperm(len(dict_ent))[:ent_per_epoch]
        epoch_start_time = time.time()
        print('-------------------')
        print('-------------------')
        print('epoch ', epoch)
        
        k = 0
        print('user_active ', len(user_active))
        user_data = []
        user_targets = []
        user_len = []
        for i in user_active:
            c += 1
            print(i)
            ue_data, ue_targets, ue_len = get_data_ue_ent_ag_classification_qs(train_data, train_target, train_len, i, num_user_train, args, seq_len,ent_active,dict_ent, qs)
            if k == 0 and ue_targets.shape[0] > 0:
                _, diff_out = train(ue_data, ue_targets, ue_len, model, result_path, epoch, i) #loss_train
                diff_out_locals = copy.deepcopy(diff_out)
                for t in diff_out_locals.keys():
                    diff_out_locals[t] = diff_out_locals[t] * w_u[i]
            elif k == 0 and ue_targets.shape[0] == 0:
                diff_out_locals = model.state_dict()
                for t in diff_out_locals.keys():
                    diff_out_locals[t] = diff_out_locals[t] * 0.0
            elif k > 0 and ue_targets.shape[0] == 0:
                diff_out = model.state_dict()
                for t in diff_out.keys():
                    diff_out[t] = diff_out[t] * 0.0
                diff_out_locals = FedAvg3(diff_out_locals, diff_out, 0.0)
            else: # k > 0 and len(train_sample_idx) > 0:
                _, diff_out = train(ue_data, ue_targets, ue_len, model, result_path, epoch, i) #loss_train
                diff_out_locals = FedAvg3(diff_out_locals, diff_out, w_u[i])
            k += 1

        diff_glob = copy.deepcopy(diff_out_locals)
        for t in diff_glob.keys():
            diff_glob[t] = torch.div(diff_glob[t], (qW*qE))
        for t in w_update.keys():
            noise = torch.zeros_like(w_update[t]).normal_(mean=0.0, std=std_**2)
            w_update[t] += diff_glob[t] + noise
        model.load_state_dict(w_update)
        #### Validation
        val_loss, val_acc = evaluate(val_data,val_target, seq_len, num_user_valid, val_len)
        print('val_loss', val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid acc {:8.2f} | valid bpc {:8.3f}'.format(
            epoch, (time.time() - epoch_start_time), val_loss, val_acc, val_loss / math.log(2)))
        print('-' * 89)

        if val_loss < stored_loss:
            model_save(args.save)
            print('Saving model (new best validation)')
            stored_loss = val_loss

        if epoch % 5 == 0 or  epoch in list_epoch or epoch == nepochs:
            if epoch in list_epoch:
                model_save(args.save[:-3] + '_epoch' + str(epoch) + '.pt')
            else:
                model_save(args.save)
            test_loss, test_acc = evaluate(test_data, test_target,seq_len, num_user_test, test_len)
            print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                    'test acc {:8.2f} | test bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), test_loss, test_acc, test_loss / math.log(2)))
            print('=' * 89)
            iter_.append(epoch)
            acc_test_all.append(val_acc)
            loss_test_all.append(test_loss)

            data_w = {'epoch': iter_, 'test acc': acc_test_all,'test loss': loss_test_all}  
            my_csv = pd.DataFrame(data_w)
            name_save = dir_path + name + '_vob{}.csv'.format(ntokens)
            my_csv.to_csv( name_save, index=False)

            if epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                print('Dividing learning rate by 10')
                optimizer2.param_groups[0]['lr'] /= 10.
            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

