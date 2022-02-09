
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
parser.add_argument('--nu', type=int, default=25,
                    help='number of users per round')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='initial learning rate')
parser.add_argument('--rs', type=int, default=0,
                    help='resume epoch')
parser.add_argument('--ns', type=float, default=2.5,
                    help='gradient clipping')
args = parser.parse_args()
args.tied = True

resume_epoch = args.rs
user_per_epoch = args.nu
S =  args.clip # 'the clip bound of the gradients'
noise_scale = args.ns


#define a batch size
batch_size = 100
lr = args.lr
# number of training epochs
# epochs = 200
md_bert = 'bert-base-uncased'                                  
max_len = 50
# clip = 0.01
is_free = 0

path = 'results/UserDP/' + md_bert + '_seq' + str(max_len) + '_bs' + str(batch_size) + '_lr' + str(lr)  + '_clip' + str(S) + '_ns' + str(noise_scale) + '_free' + str(is_free) + '_seed' + str(args.seed) + '_max0.19'

start_time = time.time()
print('batch_size', batch_size)
print('lr', lr)
# print('epochs', epochs)
print('md_bert', md_bert)
print('max_len', max_len)
print('clip', S)
print('nu', user_per_epoch)



# list_epoch = [11,176,364,552,740]#50
list_epoch = [1,32,63,94,125,156]#25, 2.5

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
    # print(classification_report(total_y, preds))
    if compute_acc == 1:
        print('Accuracy',acc)
        print('--')
    return avg_loss, acc


def FedAvg3(w_b, w_c, m):
    w_avg = deepcopy(w_b)
    for k in w_avg.keys():
        w_avg[k] = w_b[k] + w_c[k] * m
    return w_avg

def get_data_ue_ent_ag_classification_qs(i, user_, ent_active,dict_ent, ent_no_active, qs, evaluation=False):
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

# def get_data_user_ag_classification(i, user_, evaluation=False):
#     train_idx = user_[i]
#     return train_idx








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
    for i in user_active:
        c += 1
        # print('user ',i)
        train_idx = num_user_train[i]
        train_l, diff_out = train(S, train_idx, model)#loss_train
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

    diff_glob = deepcopy(diff_out_locals)
    for t in diff_glob.keys():
        diff_glob[t] = torch.div(diff_glob[t], qW)

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

