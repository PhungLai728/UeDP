
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
def train(clip):
  
  model.train()

  total_loss, total_accuracy = 0, 0
  
  # empty list to save model predictions
  total_preds=[]
  
  # iterate over batches
  for step,batch in enumerate(train_dataloader):
    
    # progress update after every 50 batches.
    if step % 500 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

    # push the batch to gpu
    batch = [r.cuda() for r in batch]
    # batch = [r for r in batch]
 
    sent_id, mask, labels = batch

    # clear previously calculated gradients 
    model.zero_grad()        

    # get model predictions for the current batch
    # print(type(sent_id))
    # print(sent_id[0])

    # print(type(mask))
    # print(mask[0])
    # exit()
    # print(type(sent_id))
    # print(type(mask))
    # print(sent_id.size())
    # print(sent_id.size())
    # print(type(labels))
    # print(labels.size())
    preds = model(sent_id, mask)
    # exit()

    # compute the loss between actual and predicted values
    loss = cross_entropy(preds, labels)
    # print('loss', loss)
    

    # add on to the total loss
    total_loss = total_loss + loss.item()

    # backward pass to calculate the gradients
    loss.backward()

    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    # update parameters
    optimizer.step()

    # model predictions are stored on GPU. So, push it to CPU
    preds=preds.detach().cpu().numpy()

    # append the model predictions
    total_preds.append(preds)

  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)
  print('total_loss', total_loss)
  
  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  #returns the loss and predictions
  return avg_loss, total_preds


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
    
    # Progress update every 50 batches.
    # if step % 50 == 0 and not step == 0:
      
      # Calculate elapsed time in minutes.
      # elapsed = format_time(time.time() - t0)
            
      # Report progress.
      # print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

    # push the batch to gpu
    batch = [t.cuda() for t in batch]
    # batch = [t for t in batch]

    sent_id, mask, labels = batch

    # deactivate autograd
    with torch.no_grad():
      
      # model predictions
      preds = model(sent_id, mask)

      # compute the validation loss between actual and predicted values
      loss = cross_entropy(preds,labels)

      total_loss = total_loss + loss.item()

      preds = preds.detach().cpu().numpy()
      # print('preds',type(preds))
      # print('labels',type(labels))
      # exit()

      total_preds.append(preds)
      total_y.append(labels)
      

  # compute the validation loss of the epoch
  avg_loss = total_loss / len(data_dataloader) 

  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)
  
  total_y = torch.cat(total_y, axis=0).cpu().numpy()
  # print('total_preds',total_preds)
  # print('total_y',total_y)
  # exit()
  if compute_acc == 0:
    return avg_loss, total_preds
  else:
    preds = np.argmax(total_preds, axis = 1)
    acc = accuracy_score(total_y, preds)
    print(classification_report(total_y, preds))
    # print('total_preds',total_preds)
    # print('preds',preds)
    # print('total_y',total_y)
    print(acc)
    # exit()
    return avg_loss, acc


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

# df = pd.read_csv("spamdata_v2.csv")
# # split train dataset into train, validation and test sets
# train_text, temp_text, train_labels, temp_labels = train_test_split(df['text'], df['label'],random_state=2018,test_size=0.3,stratify=df['label'])
# val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, random_state=2018,test_size=0.5,stratify=temp_labels)


# print(type(train_text))
# print(len(train_text))
# print(train_text[0])
# print('-1')



# print(type(train_labels))
# print(len(train_labels))
# print(train_labels[0])
# print('-3')


# exit()


#define a batch size
batch_size = 100
lr = 1e-5
# number of training epochs
epochs = 200
md_bert = 'bert-base-cased'                                  
max_len = 50
clip = 0.01
is_free = 0

path = 'results/' + md_bert + '_seq' + str(max_len) + '_bs' + str(batch_size) + '_lr' + str(lr)  + '_clip' + str(clip) + '_epochs' + str(epochs) + '_free' + str(is_free) 

start_time = time.time()
print('batch_size', batch_size)
print('lr', lr)
print('epochs', epochs)
print('md_bert', md_bert)
print('max_len', max_len)
print('clip', clip)

# import BERT-base pretrained model
bert = BertModel.from_pretrained(md_bert,return_dict=False)
# bert = BertModel.from_pretrained('bert-base-uncased')
# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(md_bert)

# tokenize and encode sequences 
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

# print(type(train_seq))
# print(type(train_mask))
# print(type(train_y))

# print(train_seq.size())
# print(train_mask.size())
# print(train_y.size())

# print(train_seq[0])
# print(train_mask[0])
# print(train_y[0])


# exit()


# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)
# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)
# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)
# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)



# wrap tensors
test_data = TensorDataset(test_seq, test_mask, test_y)
# sampler for sampling the data during training
test_sampler = RandomSampler(test_data)
# dataLoader for train set
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# freeze all the parameters
if is_free == 1:
  for param in bert.parameters():
      param.requires_grad = False


# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)
# push the model to GPU
model = model.cuda()
# model = model.to(device)


# define the optimizer
optimizer = AdamW(model.parameters(),lr = lr)          # learning rate


#compute the class weights
class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)

print("Class Weights:",class_weights)


# converting list of class weights to a tensor
weights= torch.tensor(class_weights,dtype=torch.float)

# push to GPU
weights = weights.cuda()
# weights = weights.to(device)

# define the loss function
cross_entropy  = nn.NLLLoss(weight=weights) 
# criterion = nn.BCELoss()



# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

#for each epoch
all_epoch = []
all_accuracy = []
all_loss = []
for epoch in range(epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss, _ = train(clip)
    
    #evaluate model
    valid_loss, _ = evaluate(val_dataloader, 0)
    
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), path + '.pt')
    
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

    if epoch % 10 == 0:
      loss, acc = evaluate(test_dataloader, 1)
      all_epoch.append(epoch)
      all_accuracy.append(acc)
      all_loss.append(loss)

      data_w = {'epoch': all_epoch, 'test acc': all_accuracy, 'test loss': all_loss}  
      my_csv = pd.DataFrame(data_w)
      my_csv.to_csv(path + '.csv', index=False )
      torch.save(model.state_dict(), path + '.pt')
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
print('epochs', epochs)
print('md_bert', md_bert)
print('max_len', max_len)
print('clip', clip)
print('Good job! End in ', time.time()-start_time)

