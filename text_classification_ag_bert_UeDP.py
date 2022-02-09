
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
import transformers
from transformers import AutoModel, BertTokenizerFast, BertModel
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time
import argparse
import pickle
import random
from copy import deepcopy


parser = argparse.ArgumentParser(description='PyTorch BERT AG')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
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
parser.add_argument('--clip', type=float, default=0.01,
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
user_per_epoch = args.nu
S =  args.clip # 'the clip bound of the gradients'
noise_scale = args.ns
ent_type =  args.ent

ent_percent = 0.5 #0.5
qs = args.qs


print('args.ent',args.ent)

#define a batch size
batch_size = 100
lr = args.lr
# number of training epochs
# epochs = 200
md_bert = 'bert-base-uncased'                                  
max_len = 50
# clip = 0.01
is_free = 0

path = 'results/UeDP/' + md_bert + '_seq' + str(max_len) + '_bs' + str(batch_size) + '_lr' + str(lr)  + '_clip' + str(S) + '_e' + ent_type + '_ep' + str(ent_percent) + '_ns' + str(noise_scale) + '_free' + str(is_free) + '_seed' + str(args.seed) + '_qs' + str(args.qs) + '_max0.19'

start_time = time.time()
print('batch_size', batch_size)
print('lr', lr)
# print('epochs', epochs)
print('md_bert', md_bert)
print('max_len', max_len)
print('clip', S)
print('nu', user_per_epoch)
print('ent', ent_type)
print('qs', qs)



if ent_type == 'gpe':
    # list_epoch = [236,383, 537, 711, 907,1125]#,4513]#,1151,1154,1649,1650,1651,1654] # 1000
    # [0.5,0.6,0.7,0.8,0.9,1,2]
    # list_epoch = [13,180,372,563,755] #600 # Bert [0.3,0.35,0.4,0.45,0.5]
    list_epoch = [1,33]#,64,96,127,159] #300, 2.5 # Bert [0.18,0.182,...,0.19]

elif ent_type == 'org':
    # list_epoch = [73,137,201,270,347,434]#,1786] #500 [0.5,0.6,0.7,0.8,0.9,1,2]
    # list_epoch = [5,160,331,503,677] #200 # Bert [0.3,0.35,0.4,0.45,0.5]
    list_epoch = [1,30]#,58,87,116,144] #100, 2.5 # Bert [0.18,0.182,...,0.19]

elif ent_type == 'loc':
    # list_epoch = [195,324,456,603,771,958]#,3854] #500 [0.5,0.6,0.7,0.8,0.9,1,2]
    # list_epoch = [31,253,484,716,947] #250 # Bert [0.3,0.35,0.4,0.45,0.5]
    list_epoch = [1,38]#,74,110,146,182] #130, 2.5 # Bert [0.18,0.182,...,0.19]

# elif ent_type == 'all':
#     list_epoch = [51,100,150,203,262,329]#,1367] #500 [0.5,0.6,0.7,0.8,0.9,1,2]
elif ent_type == 'pii':
    # list_epoch = [167,279,396,525,672,836]#,3373] #500
    # list_epoch = [18,206,405,614,823] #250 # Bert [0.3,0.35,0.4,0.45,0.5]
    list_epoch = [1,33]#,64,96,127,159] #130, 2.5 # Bert [0.18,0.182,...,0.19]

elif ent_type == 'allpii':
    # list_epoch = [49,97,145,254,319,1327] #500  
    # list_epoch = [26,248,476,703,930] #150 # Bert [0.3,0.35,0.4,0.45,0.5] 
    # list_epoch = [4,156,322,488,663] #175 # Bert [0.3,0.35,0.4,0.45,0.5] 
    list_epoch = [1,22]#,43,65,86,107] #100, 2.5 # Bert [0.18,0.182,...,0.19]

nepochs = list_epoch[-1] #37217 # 335

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)






class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      self.dropout = nn.Dropout(0.1)
      self.relu =  nn.ReLU()
      self.fc1 = nn.Linear(768,512)
      self.fc2 = nn.Linear(512,4)
      self.softmax = nn.LogSoftmax(dim=1)
      # self.softmax = nn.Sigmoid()

    #define the forward pass
    def forward(self, sent_id, mask):
      _, cls_hs = self.bert(sent_id, attention_mask=mask)
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)
      x = self.softmax(x)
      return x



# function to train the model
def train(S, train_idx, model):
    # print('model.state_dict() before train', model.state_dict()['fc2.bias'])
    
    model_local.load_state_dict(model.state_dict(), strict=False)
    # local_before =  deepcopy(model_local.state_dict())
    # print('local_before before train', local_before['fc2.bias'])

    model_local.train()

    total_loss, total_accuracy = 0, 0
  
    # empty list to save model predictions
    # total_preds=[]
  
    # iterate over batches
    bs = 10
    # print('len(train_idx)',len(train_idx))
    if len(train_idx) < bs + 1:
        no_iter = 1
    elif len(train_idx)%bs==0:
        no_iter = len(train_idx)//bs
    else:
        no_iter = len(train_idx)//bs + 1

    for i in range(no_iter):
        if i == no_iter-1:
            start_ = i*bs
            end_ = len(train_idx)
        else:
            start_ = i*bs
            end_ = (i+1)*bs



    # for step in train_idx: #range(len((train_data))):
        # for step,batch in enumerate(train_dataloader):
        # push the batch to gpu
        batch = train_data[train_idx[start_:end_]]
        batch = [r.cuda() for r in batch]
        # batch = [r for r in batch]
 
        sent_id, mask, labels = batch
        # sent_id = torch.unsqueeze(sent_id,0) 
        # mask = torch.unsqueeze(mask,0)  
        # labels = torch.unsqueeze(labels,0) 
        # print('sent_id',sent_id)
        # print('mask',mask)
        # print('labels',labels)
        # print('sent_id',sent_id.size())
        # exit()

        # clear previously calculated gradients 
        model_local.zero_grad()        
        preds = model_local(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy_local(preds, labels)
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_local.parameters(), S)
        optimizer_local.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()

        # append the model predictions
        # total_preds.append(preds)
    # exit()
    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_idx)
    # print('total_loss', total_loss)
    # print('avg_loss', avg_loss)

    global_w = deepcopy(model.state_dict())
    local_w =  deepcopy(model_local.state_dict())
    differ_w = deepcopy(model.state_dict())

    for k in differ_w.keys():
        differ_w[k] = local_w[k] - global_w[k]
  
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    # total_preds  = np.concatenate(total_preds, axis=0)

    #returns the loss and predictions
    return avg_loss, differ_w#, total_preds


  # function for evaluating the model
def evaluate(data_dataloader, compute_acc):
  
    print("\nEvaluating...")
  
    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0
  
    # empty list to save the model predictions
    total_preds = []
    total_y = []

    # iterate over batches
    for step,batch in enumerate(data_dataloader):

        # push the batch to gpu
        batch = [t.cuda() for t in batch]
        # batch = [t for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():
      
            # model predictions
            preds = model(sent_id, mask)
            loss = cross_entropy(preds,labels)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)
            total_y.append(labels)
      
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(data_dataloader) 

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)
    total_y = torch.cat(total_y, axis=0).cpu().numpy()

  # if compute_acc == 0:
  #   return avg_loss, total_preds
  # else:
    preds = np.argmax(total_preds, axis = 1)
    acc = accuracy_score(total_y, preds)
    # 
    if compute_acc == 1:
        print('--')
        print(classification_report(total_y, preds))
        print('Accuracy ',acc)
        print('--**--')
    return avg_loss, acc


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










# specify GPU
# device = torch.device("cuda")

df = pd.read_csv("../../data/train_preprocess.csv")
train_text = df['Description-ori']
train_labels = df['Class Index']

df = pd.read_csv("../../data/valid_preprocess.csv")
val_text = df['Description-ori']
val_labels = df['Class Index']

df = pd.read_csv("../../data/test_preprocess.csv")
test_text = df['Description-ori']
test_labels = df['Class Index']
# print('train', len(train_text)) # 112,000
# print('test', len(test_text))
# print('val', len(val_text))

# import BERT-base pretrained model
bert = BertModel.from_pretrained(md_bert,return_dict=False)
# bert = BertModel.from_pretrained('bert-base-uncased')
# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(md_bert)

# freeze all the parameters
if is_free == 1:
    for param in bert.parameters():
        param.requires_grad = False

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)
model_local = BERT_Arch(bert)
# push the model to GPU
model = model.cuda()
model_local = model_local.cuda()
# model = model.to(device)

# define the optimizer
optimizer = AdamW(model.parameters(),lr = lr)          # learning rate
optimizer_local = AdamW(model_local.parameters(),lr = lr)          # learning rate


#compute the class weights
class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)

# print("Class Weights:",class_weights)


# converting list of class weights to a tensor
weights= torch.tensor(class_weights,dtype=torch.float)

# push to GPU
weights = weights.cuda()
# weights = weights.to(device)

# define the loss function
cross_entropy  = nn.NLLLoss(weight=weights) 
cross_entropy_local  = nn.NLLLoss(weight=weights) 

# criterion = nn.BCELoss()

tmp = np.load('../../data/trainUserDataCount_AG2.npz',allow_pickle=True)
num_user_train = tmp['user_idx']
tmp = np.load('../../data/testUserDataCount_AG2.npz',allow_pickle=True)
num_user_test = tmp['user_idx']
tmp = np.load('../../data/validUserDataCount_AG2.npz',allow_pickle=True)
num_user_valid = tmp['user_idx']
flat_list = [item for sublist in num_user_train for item in sublist]
# print(max(flat_list))
# print(min(flat_list))
# print(len(flat_list))


all_sent_dicts =pickle.load(open("../../data/ag_sensitive_dicts", "rb"))  
# ent_type = 'all'
# all_sent_dicts = [dict_org, dict_person,dict_loc,dict_gpe,dict_product]

if ent_type == 'loc':
    dict_ent = all_sent_dicts[2]
elif ent_type == 'person':
    dict_ent = all_sent_dicts[1]
elif ent_type == 'gpe':
    dict_ent = all_sent_dicts[3]
elif ent_type == 'org':
    dict_ent = all_sent_dicts[0]
elif ent_type == 'pii':
    dict_1 = all_sent_dicts[1]
    dict_2 = all_sent_dicts[2]
    dict_ent = {}
    for d in [dict_2, dict_3]:
      dict_ent.update(d)
else:
    dict_0 = all_sent_dicts[0]
    dict_1 = all_sent_dicts[1]
    dict_2 = all_sent_dicts[2]
    dict_3 = all_sent_dicts[3]

    dict_ent = {}
    for d in [dict_0, dict_2, dict_3]:
      dict_ent.update(d)

num_user = len(num_user_train) # 'number of users N1'
print('num_user', num_user)
# print('73 ', num_user_train[73])
sr_user = user_per_epoch/num_user



tmp = np.load('../../data/ag_' + ent_type + '_sens_indicator.npz',allow_pickle=True)
sens_indicator = tmp['sens_indicator']


data4ue = np.load('../../data/train_ag_ue_30k_seq30.npz',allow_pickle=True)
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
#     non_ent_list = [a for a in non_ent if a not in  loc_list and a not in gpe_list and a not in product_list ]
        # and a not in org_list and a not in date_list and a not in person_list and a not in fac_list]

sr_entity = ent_percent #ent_per_epoch/num_entity # e.g., 0.5
ent_per_epoch = int(ent_percent* len(dict_ent))

# tmp = np.load('non_ent.npz',allow_pickle=True)
# non_ent_list = tmp['non_ent_list']
num_noEntity = 112000 - num_entity# 'number of entities N2'
# num_noEntity = 78431
sr_non_entity = 1
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

# exit()








# Data tokenize and encode sequences 
tokens_train = tokenizer.batch_encode_plus(train_text.tolist(),max_length = max_len,pad_to_max_length=True,truncation=True)
tokens_val = tokenizer.batch_encode_plus(val_text.tolist(), max_length = max_len, pad_to_max_length=True,truncation=True)
tokens_test = tokenizer.batch_encode_plus(test_text.tolist(),max_length = max_len,pad_to_max_length=True,truncation=True)

## convert lists to tensors
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

train_data = TensorDataset(train_seq, train_mask, train_y)
val_data = TensorDataset(val_seq, val_mask, val_y)
test_data = TensorDataset(test_seq, test_mask, test_y)
# train_dataloader = DataLoader(train_data, sampler=None, batch_size=batch_size)
val_dataloader = DataLoader(val_data, sampler = None, batch_size=batch_size)
test_dataloader = DataLoader(test_data, sampler=None, batch_size=batch_size)








# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

#for each epoch
all_epoch = []
all_accuracy = []
all_loss = []
all_accuracy_val = []
c = 0
for epoch in range(nepochs + 1):

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
    print('user_active ', len(user_active))
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
        # # print((train_idx))
        # print(k)
        # exit()


        if k == 0 and len(train_idx) > 0:
            train_l, diff_out = train(S, train_idx, model)
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
            train_l, diff_out = train(S, train_idx, model)
            diff_out_locals = FedAvg3(diff_out_locals, diff_out, w_u[i])
        # print('train_l',train_l)
        # print('--')

        train_loss += train_l
        k += 1

    print('Done each epoch in ', time.time()-epoch_start_time)


    train_loss = train_loss/len(user_active)

    diff_glob = deepcopy(diff_out_locals)
    for t in diff_glob.keys():
        diff_glob[t] = torch.div(diff_glob[t], (qW*qE))

    for t in w_update.keys():
        noise = torch.empty(w_update[t].size()).normal_(mean=0.0, std=std_**2).cuda()
        w_update[t] += diff_glob[t] + noise

    model.load_state_dict(w_update)

    #evaluate model
    valid_loss, valid_acc = evaluate(val_dataloader,0)
    print('valid_loss',valid_loss)
    print('best_valid_loss',best_valid_loss)
    # exit()
    
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), path + '.pt')
        print('Good model! Saving...')
    
    

    if epoch % 5 == 0 or  epoch in list_epoch or epoch == nepochs:
        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        all_accuracy_val.append(valid_acc)
        
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

        loss, acc = evaluate(test_dataloader,1)
        all_epoch.append(epoch)
        all_accuracy.append(acc)
        all_loss.append(loss)

        data_w = {'epoch': all_epoch, 'test acc': all_accuracy, 'valid acc': all_accuracy_val, 'test loss': all_loss, 'train loss': train_losses, 'val loss': valid_losses}  
        my_csv = pd.DataFrame(data_w)
        my_csv.to_csv(path + '.csv', index=False )
        torch.save(model.state_dict(), path + '.pt')
      # exit()
    #   # get predictions for test data
    #   with torch.no_grad():
    #     # preds = model(test_seq, test_mask)
    #     preds = model(test_seq.cuda(), test_mask.cuda())
    #     preds = preds.detach().cpu().numpy()

    #   preds = np.argmax(preds, axis = 1)
    #   print(classification_report(test_y, preds))



#load weights of best model
model.load_state_dict(torch.load(path + '.pt'))

# test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size=batch_size)
evaluate(test_dataloader, 1)

# # get predictions for test data
# test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size=batch_size)

# with torch.no_grad():
#   # preds = model(test_seq, test_mask)
#   preds = model(test_seq.cuda(), test_mask.cuda())
#   preds = preds.detach().cpu().numpy()

# preds = np.argmax(preds, axis = 1)
# print(classification_report(test_y, preds))


print('batch_size', batch_size)
print('lr', lr)
print('nepochs', nepochs)
print('md_bert', md_bert)
print('max_len', max_len)
print('clip', S)
print('Good job! End in ', time.time()-start_time)

