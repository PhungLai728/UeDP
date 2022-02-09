import torch
import numpy as np
import itertools
import random

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, labels, data_len, bsz, args, seq_len):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(data) // (bsz * seq_len)
    n = len(data) // (bsz * seq_len) * (bsz * seq_len)
    n1 = len(labels) // (bsz) *bsz
    
    data = torch.LongTensor(data[:n])
    # print(data[:2*30])
    # exit()
    labels = torch.LongTensor(labels[:n1])
    data_len = torch.LongTensor(data_len[:n1])

    data = data.narrow(0, 0, nbatch * bsz * seq_len)
    data = data.view(bsz, seq_len * nbatch).t().contiguous()
    data_len = data_len.view(bsz, nbatch).t().contiguous()
    labels = labels.view(bsz, nbatch).t().contiguous()
    if args.cuda: 
        data = data.cuda()
        labels = labels.cuda()
    return data, labels, data_len

def batchify_up(data, labels, data_len, bsz, args, seq_len, up):
    # up = 0
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(data) // (bsz * seq_len)
    n = len(data) // (bsz * seq_len) * (bsz * seq_len)
    n1 = len(labels) // (bsz) *bsz
    # print(nbatch)
    # print(bsz)
    # print(seq_len)
    # print('data', data.size())
    data = torch.LongTensor(data[:n])
    labels = torch.LongTensor(labels[:n1])
    data_len = torch.LongTensor(data_len[:n1])
    data = data.narrow(0, 0, nbatch * bsz * seq_len)

    # print('data', data.size())
    # exit()
    # up = 3
    label1 = (labels == 1).nonzero()
    label1 =  torch.squeeze(label1)
    # print(label1)
    # print(label1.size())
    nadd = len(label1) // (bsz) *bsz
    label1 = label1[:nadd]
    # print(label1)
    # print(label1.size())
    # exit()

    data_add = torch.LongTensor(up*seq_len*len(label1))
    
    labels_up  = torch.LongTensor(len(labels) + up*len(label1))
    for i in range(len(labels_up)):
        if i < len(labels):
            labels_up[i] = labels[i]
        else:
            labels_up[i] = 1

    data_len_add = torch.LongTensor(up*len(label1))
    for k in range(up):
        for i in range(len(label1)):
            # data_len =  torch.cat((data_len, torch.unsqueeze(data_len[label1[i]],0)))    
            data_len_add[k*len(label1) + i] = data_len[label1[i]]
    data_len = torch.cat((data_len, data_len_add))    

    for k in range(up):
        for i in range(len(label1)):
            data =  torch.cat((data,  data[seq_len*label1[i]: seq_len*(label1[i]+1)])) 
            
    #         data_add[k*len(label1) + i : k*len(label1) + i] = data[seq_len*label1[i]: seq_len*(label1[i]+1)]
    # data_up = torch.cat((data, data_add)) 

    # data = data.repeat(up)
    # data_len = data_len.repeat(up)
    # labels = labels.repeat(up) 
    

    data = data.view(bsz,-1 ).t().contiguous() #up*seq_len * nbatch
    # print('data',data.size())
    data_len = data_len.view(bsz, -1).t().contiguous() #up*nbatch
    # print('data_len',data_len.size())
    labels_up = labels_up.view(bsz, -1).t().contiguous()#up*nbatch
    # print('labels_up',labels_up.size())
    # exit()
    if args.cuda: 
        data = data.cuda()
        labels_up = labels_up.cuda()
    return data, labels_up, data_len

# def get_batch(source, labels, len_, i, args, seq_len=None, evaluation=False):
#     # print(i)
#     # print(i*(seq_len+1))
#     # print((i+1)*(seq_len+1)-1)
#     # print(i*(seq_len+1)+1)
#     # print((i+1)*(seq_len+1))
#     # seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
#     data = source[i*seq_len:(i+1)*seq_len]
#     target = [labels[i]]
#     len_sm = [len_[i]]
#     # seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
#     # data = source[i:i+seq_len]
#     # target = source[i+1:i+1+seq_len]
#     # target = list(itertools.chain(*target))
#     data = torch.LongTensor(data)
#     len_sm = torch.LongTensor(len_sm)
#     target = torch.LongTensor(target)
#     # print(type(len_sm))
#     # print(len_sm)
#     # exit()
#     if args.cuda: 
#         data = data.cuda()
#         target = target.cuda()
#         # len_sm = len_sm.cuda()
#     return data, target, len_sm

def get_batch(source, labels, len_, i, args, seq_len=None, evaluation=False):
    # print(i)
    # print(i*(seq_len+1))
    # print((i+1)*(seq_len+1)-1)
    # print(i*(seq_len+1)+1)
    # print((i+1)*(seq_len+1))
    # seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i*seq_len:(i+1)*seq_len]
    target = labels[i]
    len_sm = len_[i]
    # print(type(data))
    # print(type(target))
    # print(type(len_sm))
    # print(data.size())
    # print(target.size())
    # print(len_sm.size())
    # exit()
    

    # seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    # data = source[i:i+seq_len]
    # target = source[i+1:i+1+seq_len]
    # target = list(itertools.chain(*target))
    return data, target, len_sm


def batchify_lm(data, data_len, bsz, args, seq_len):
    # # Work out how cleanly we can divide the dataset into bsz parts.
    # nbatch = len(data) // (bsz * seq_len)
    # data = data.narrow(0, 0, nbatch * bsz * seq_len)
    # data = data.view(bsz, seq_len * nbatch).t().contiguous()
    # data_len = data_len.view(bsz, nbatch).t().contiguous()
    # if args.cuda: 
    #     data = data.cuda()
    #     # labels = labels.cuda()
    # return data, data_len
    nbatch = len(data) // (bsz * (seq_len+1))
    data = data.narrow(0, 0, nbatch * bsz * (seq_len+1))
    data = data.view(bsz, (seq_len+1) * nbatch).t().contiguous()
    data_len = data_len.view(bsz, nbatch).t().contiguous()
    if args.cuda: 
        data = data.cuda()
    return data, data_len

def get_batch_lm(source, len_, i, args, seq_len=None, evaluation=False):
    data = source[i*(seq_len+1):(i+1)*(seq_len+1)-1]
    target = source[(i+1)*(seq_len+1)-1].view(-1)
    len_sm = len_[i]
    # data = source[i:i+seq_len]
    # target = source[i+1:i+1+seq_len].view(-1)
    return data, target, len_sm


def get_data_user_conll(source, len_, i, user_, args, seq_len, evaluation=False):
    cumsum_ = np.cumsum(user_)
    if i == 0:
        num_sent = cumsum_[i]
        start_w = 0
        end_w = cumsum_[i]*(seq_len+1)
        len_sm = len_[:cumsum_[i]]
    else:
        num_sent = cumsum_[i] - cumsum_[i-1]
        start_w = cumsum_[i-1]*(seq_len+1)
        end_w = cumsum_[i]*(seq_len+1) # From start to end-1
        len_sm = len_[cumsum_[i-1]: cumsum_[i]]
    # print(start_w)
    # print(end_w)
    # exit()

    source = torch.LongTensor(source)
    len_sm = torch.LongTensor(len_sm)
    all_ = source[start_w:end_w].view(num_sent, seq_len+1).t().contiguous()
    data = all_[:seq_len] # from 0 to seq_len -1
    target = all_[seq_len].view(-1) # the last one
    if args.cuda: 
        data = data.cuda()
        target = target.cuda()
        len_sm = len_sm.cuda()
    return data, target, len_sm

def get_data_ue_ent_ag(source, len_, i, user_, args, seq_len,ent_active,dict_ent,  evaluation=False):
    # print('source',len(source))
    # # print('target_',len(target_))
    # print('len_',len(len_))
    # print('i',i)
    # print('user_',len(user_))
    # # print(target_)
    # print(user_)
    # print(len_)
    # print(source)
    # exit()
    have_sent = []
    for e in ent_active:
        if e.item() in dict_ent:
            # print(i.item())
            # print(type(i.item()))
            tmp = dict_ent[e.item()]
            have_sent.extend(tmp)
    # print('have_sent', have_sent)

    samples = user_[i]
    # print('samples', samples)
    # exit()
    train_idx = [s for s in samples if s not in have_sent]
    num_sent = len(train_idx)
    for_train = [list(range(t*(seq_len+1),(t+1)*(seq_len+1))) for t in train_idx]
    flat_list = [item for sublist in for_train for item in sublist]
    len_sm = len_[train_idx]
    # # print(len(flat_list))
    # print('train_idx',train_idx)
    # print('flat_list',flat_list)
    # print('source[flat_list]', source[flat_list[:5]])
    # print('source', source[:10])
    # print('source', source[[0,1,4,5]])
    # # print('target',target)
    # exit()

    source = torch.LongTensor(source)
    len_sm = torch.LongTensor(len_sm)
    # all_ = source[start_w:end_w].view(num_sent, seq_len+1).t().contiguous()
    all_ = source[flat_list].view(num_sent, seq_len+1).t().contiguous()
    # print(source[:35])
    # exit()
    # print('all_',len(all_))
    # print('all_',source[flat_list].view(num_sent, seq_len+1).shape)
    # print('all_',source[flat_list].view(num_sent, seq_len+1).t().shape)
    # print('all_',all_)
    # exit()

    data = all_[:seq_len] # from 0 to seq_len -1
    target = all_[seq_len].view(-1) # the last one
    # print('data', data.shape)
    # print(data[:2])
    # print('target', target.shape)
    # print(target)
    # print(len_sm)
    # exit()
    
    if args.cuda: 
        data = data.cuda()
        target = target.cuda()
        len_sm = len_sm.cuda()
    return data, target, len_sm


def get_data_ue_ent_ag_classification(source, target_, len_, i, user_, args, seq_len,ent_active,dict_ent,  evaluation=False):
    # print('source',len(source))
    # print('target_',len(target_))
    # print('len_',len(len_))
    # print('i',i)
    # print('user_',len(user_))
    # print(target_)
    # print(user_)
    # print(len_)
    # print(source)
    # exit()
    have_sent = []
    for e in ent_active:
        if e.item() in dict_ent:
            # print(i.item())
            # print(type(i.item()))
            tmp = dict_ent[e.item()]
            have_sent.extend(tmp)
    # print('have_sent', have_sent)

    samples = user_[i]
    # print('samples', samples)
    # exit()
    train_idx = [s for s in samples if s not in have_sent]
    num_sent = len(train_idx)
    for_train = [list(range(t*(seq_len),(t+1)*(seq_len))) for t in train_idx]
    flat_list = [item for sublist in for_train for item in sublist]
    # print(flat_list)
    # print('train_idx',train_idx)

    len_sm = len_[train_idx]
    target = target_[train_idx]
    # print('len_sm',len_sm)
    # print('target',target)
    # exit()

    source = torch.LongTensor(source)
    len_sm = torch.LongTensor(len_sm)
    target = torch.LongTensor(target)
    all_ = source[flat_list].view(num_sent, seq_len).t().contiguous()
    # print(source[:35])
    # exit()

    data = all_[:seq_len] # from 0 to seq_len -1
    # target = all_[seq_len].view(-1) # the last one
    # print('data', data.shape)
    # print(data[:2])
    # print('target', target.shape)
    # print(target)
    # print(len_sm)
    # exit()
    if args.cuda: 
        data = data.cuda()
        target = target.cuda()
        len_sm = len_sm.cuda()
    return data, target, len_sm

def get_data_ue_ent_ag_classification_qs(source, target_, len_, i, user_, args, seq_len,ent_active,dict_ent, qs, evaluation=False):
    have_sent = []
    for e in ent_active:
        if e.item() in dict_ent:
            # print(i.item())
            # print(type(i.item()))
            tmp = dict_ent[e.item()]
            have_sent.extend(tmp)

    samples = user_[i]
    # print('samples',samples)

    train_idx = [s for s in samples if s not in have_sent]
    # in_have_sent = [s for s in samples if s in have_sent]
    # if len(in_have_sent) == 0: # all are non-sensitive
    n = int(len(train_idx)*qs)
    train_idx = random.sample(list(train_idx), n)
    # else:
    # print('train_idx',train_idx)
    # print(len(train_idx))
    
    # exit()


    num_sent = len(train_idx)
    for_train = [list(range(t*(seq_len),(t+1)*(seq_len))) for t in train_idx]
    flat_list = [item for sublist in for_train for item in sublist]

    len_sm = len_[train_idx]
    target = target_[train_idx]

    source = torch.LongTensor(source)
    len_sm = torch.LongTensor(len_sm)
    target = torch.LongTensor(target)
    all_ = source[flat_list].view(num_sent, seq_len).t().contiguous()

    data = all_[:seq_len] # from 0 to seq_len -1
    if args.cuda: 
        data = data.cuda()
        target = target.cuda()
        len_sm = len_sm.cuda()
    # exit()
    return data, target, len_sm

def get_data_user(source, len_, i, user_, args, seq_len, evaluation=False):
    samples = user_[i]
    all_ = []
    len_sm = []
    for j in range(len(samples)):
        idx = samples[j]
        tmp =  source[idx*(seq_len+1):(idx+1)*(seq_len+1)]
        all_.append(tmp)
        len_sm.append(len_[idx])
    all_ = torch.LongTensor(all_)
    data = all_.t()[:seq_len] # from 0 to seq_len -1
    target = all_.t()[seq_len] # the last one
    len_sm = torch.LongTensor(len_sm)
    
    if args.cuda: 
        data = data.cuda()
        target = target.cuda()
        len_sm = len_sm.cuda()
    return data, target, len_sm

def get_data_user_classification(source, target_, len_, i, user_, args, seq_len, evaluation=False):
    samples = user_[i]
    all_ = []
    len_sm = []
    target = []
    # print(i)
    # print(type(samples))
    # print(samples)
    # exit()
    for j in range(len(samples)):
        idx = samples[j]
        tmp =  source[idx*(seq_len):(idx+1)*(seq_len)]
        all_.append(tmp)
        len_sm.append(len_[idx])
        target.append(target_[idx])
    all_ = torch.LongTensor(all_)
    # print('all_', all_.size())
    data = all_.t()[:seq_len] # from 0 to seq_len -1
    # print('data',data.size())
    # target = all_.t()[seq_len] # the last one
    len_sm = torch.LongTensor(len_sm)
    target = torch.LongTensor(target)
    # print('len_sm',len_sm.size())
    # print('target',target.size())
    # exit()
    
    if args.cuda: 
        data = data.cuda()
        target = target.cuda()
        len_sm = len_sm.cuda()
    return data, target, len_sm

def get_data_ue_ent_conll(source, len_, i, user_, args, seq_len,ent_active,dict_ent, evaluation=False):

    have_sent = []
    for e in ent_active:
        if e.item() in dict_ent:
            # print(i.item())
            # print(type(i.item()))
            tmp = dict_ent[e.item()]
            have_sent.extend(tmp)
    #     else:
    #         print('lolo')
    # print(have_sent)
    # exit()

    # i = 0

    cumsum_ = np.cumsum(user_)
    if i == 0:
        # num_sent = cumsum_[i]
        # print('num_sent',num_sent)
        # start_w = 0
        # end_w = cumsum_[i]*(seq_len+1)
        len_sm = len_[:cumsum_[i]]
        start_idx = 0
        end_idx = cumsum_[i]
    else:
        # num_sent = cumsum_[i] - cumsum_[i-1]
        # print('num_sent',num_sent)
        # start_w = cumsum_[i-1]*(seq_len+1)
        # end_w = cumsum_[i]*(seq_len+1) # From start to end-1
        len_sm = len_[cumsum_[i-1]: cumsum_[i]]
        start_idx = cumsum_[i-1]
        end_idx = cumsum_[i]

    total_s = list(range(start_idx, end_idx))
    train_idx = [s for s in total_s if s not in have_sent]
    num_sent = len(train_idx)
    for_train = [list(range(t*(seq_len+1),(t+1)*(seq_len+1))) for t in train_idx]
    flat_list = [item for sublist in for_train for item in sublist]
    len_sm = len_[train_idx]



    # print('i',i)
    # print('start_idx',start_idx)
    # print('end_idx',end_idx)
    # print('num_sent',num_sent)
    # print('train_idx',train_idx)
    # print('for_train',for_train)
    # print('flat_list',flat_list)
    # exit()

    source = torch.LongTensor(source)
    len_sm = torch.LongTensor(len_sm)
    # all_ = source[start_w:end_w].view(num_sent, seq_len+1).t().contiguous()
    all_ = source[flat_list].view(num_sent, seq_len+1).t().contiguous()
    # print('all_',len(all_))
    # print('all_',all_.shape)
    # print('all_',all_)
    # # exit()

    data = all_[:seq_len] # from 0 to seq_len -1
    target = all_[seq_len].view(-1) # the last one

    if args.cuda: 
        data = data.cuda()
        target = target.cuda()
        len_sm = len_sm.cuda()
    # print('data',data.shape)
    # print('data',data)
    # print('target',target.shape)
    # print('target',target)
    # print('len_sm',len_sm.shape)
    # print('len_sm',len_sm)
    # exit()
    return data, target, len_sm


def get_batch_ue_eval(source, len_, i, user_, args, seq_len=None, evaluation=True):
    cumsum_ = np.cumsum(user_)
    if i == 0:
        num_sent = cumsum_[i]
        start_w = 0
        end_w = cumsum_[i]*(seq_len+1)
        len_sm = len_[:cumsum_[i]]
    else:
        num_sent = cumsum_[i] - cumsum_[i-1]
        start_w = cumsum_[i-1]*(seq_len+1)
        end_w = cumsum_[i]*(seq_len+1) # From start to end-1
        len_sm = len_[cumsum_[i-1]: cumsum_[i]]

    source = torch.LongTensor(source)
    len_sm = torch.LongTensor(len_sm)

    all_ = source[start_w:end_w].view(num_sent, seq_len+1).t().contiguous()

    data = all_[:seq_len] # from 0 to seq_len -1
    target = all_[seq_len].view(-1) # the last one
    
    if args.cuda: 
        data = data.cuda()
        target = target.cuda()
        len_sm = len_sm.cuda()
    return data, target, len_sm