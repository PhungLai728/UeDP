
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


def choose_from_top(probs, n=1):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

class SEC_data(Dataset):
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



# all_info = pd.read_csv('../../data/SEC_v5/data/train.csv')
# sec = all_info['text']
# len_ = []
# for i in range(len(sec)):
#     len_.append(len(sec[i].split()))
# print(min(len_)) #1
# print(max(len_)) #1
# print(sum(len_)/len(len_)) #124.65381649961449
# l256 = [1 for i in len_ if i >256]
# print(sum(l256)) #808
# print(len(sec)) #5188
# exit()



BATCH_SIZE = 1000
EPOCHS = 5
LEARNING_RATE = 1e-3
WARMUP_STEPS = 5000 #50
# MAX_SEQ_LEN = 50
model_type = "gpt2"
max_ = 128#256#354
S = 0.01
path_csv = 'results_nwp/conll/' + model_type + '_bs' + str(BATCH_SIZE)  + '_wt' + str(WARMUP_STEPS) + '_lr' + str(LEARNING_RATE) + '_max' + str(max_) + '_clip' + str(S)

print('LEARNING_RATE', LEARNING_RATE)
print('max_', max_)
print('clip',S)
print('path_csv',path_csv)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

start_time = time.time()
train_data = SEC_data('train_conll.csv', model_type, max_) 
# val_data = AG_data('valid_preprocess.csv', model_type, 1024) 
test_data = SEC_data('test_conll.csv', model_type, max_) 
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
print('--- Load model in ',  time.time()-start_time)
for param in model.parameters():
    param.requires_grad = True

model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps = -1)


# models_folder = "trained_models"
# if not os.path.exists(models_folder):
#     os.mkdir(models_folder)

train_losses=[]
train_ppl=[]
test_losses=[]
test_ppl=[]
all_iters = []
iter_ = 0
for epoch in range(EPOCHS):
    
    print(f"EPOCH {epoch} started" + '=' * 30)
    # c = 0
    sum_loss = 0.0
    tmp_seq = None

    # for idx,joke in enumerate(train_loader):
        # print('sample ', c)
    for ttt in range(len(train_data)): 
        
        
        #################### "Fit as many joke sequences into MAX_SEQ_LEN sequence as possible" logic start ####
        train_data_seq = train_data[ttt].unsqueeze(0).to(device)
        # print('seq',seq)
        

        # #Skip sample from dataset if it is longer than MAX_SEQ_LEN
        # if seq.size()[1] > MAX_SEQ_LEN:
        #     continue
        
        # #The first data sequence in the sequence
        # if not torch.is_tensor(tmp_seq):
        #     tmp_seq = deepcopy(seq)
        #     continue
        # else:
        #     #The next joke does not fit in so we process the sequence and leave the last joke 
        #     #as the start for next sequence 
        #     if tmp_seq.size()[1] + seq.size()[1] > MAX_SEQ_LEN:
        #         train_data_seq = deepcopy(tmp_seq)
        #         tmp_seq = deepcopy(seq)
        #     else:
        #         #Add the joke to sequence, continue and try to add more
        #         tmp_seq = torch.cat([tmp_seq, seq[:,1:]], dim=1)
        #         continue
        ################## Sequence ready, process it through the model ##################
        # train_data_seq = train_data_seq.to(device)
        outputs = model(train_data_seq, labels=train_data_seq)
        # outputs=model(train_data_seq[:-1], labels=train_data_seq[1:])
        # outputs=model(train_data_seq)

        loss, logits = outputs[:2] #outputs[:2] = outputs[0]
        loss.backward()
        sum_loss += loss.detach().data
        # print('proc_seq_count',proc_seq_count)
        # exit()
        iter_ += 1  
        if (iter_-1) % BATCH_SIZE == 0:
            # proc_seq_count = 0    
            # batch_count += 1
            print('iter_ ', iter_)
            torch.nn.utils.clip_grad_norm_(model.parameters(), S)
            optimizer.step()
            # scheduler.step() 
            optimizer.zero_grad()
            model.zero_grad()

            if (iter_-1) == 0:
                loss_train = deepcopy(sum_loss)
            else:
                loss_train = sum_loss/BATCH_SIZE
            
                sum_loss = 0.0

            print(f"sum loss {loss_train*BATCH_SIZE}")
            print(f"ppl {torch.exp(loss_train)}")
            # print(f"ppl {math.exp(loss_train)}")
        if (iter_-1) % 1000 == 0: 
            loss_test, ppl_test = evaluate(test_data)  
            print(f"sum loss test {loss_test}")
            print(f"ppl {ppl_test}")
            print('---')

            train_losses.append(loss_train.to('cpu').numpy())
            train_ppl.append(torch.exp(loss_train).to('cpu').numpy())
            all_iters.append(iter_)
            test_losses.append(loss_test.to('cpu').numpy())
            test_ppl.append(ppl_test.to('cpu').numpy())

            data_w = {'iters': all_iters, 'test ppl': test_ppl, 'test loss': test_losses, 'train loss': train_losses, 'train ppl': train_ppl}  
            my_csv = pd.DataFrame(data_w)
            my_csv.to_csv(path_csv + '.csv', index=False )
            torch.save(model.state_dict(), os.path.join('', f"" + path_csv + ".pt") )

        # c += 1 
        
# Store the model after each epoch to compare the performance of them
torch.save(model.state_dict(), os.path.join('', f"" + path_csv + ".pt") )
print('LEARNING_RATE', LEARNING_RATE)
print('max_', max_)
print('path_csv',path_csv)

exit()
















MODEL_EPOCH = 1

models_folder = "trained_models"

model_path = os.path.join(models_folder, f"gpt2_joker_{MODEL_EPOCH}.pt")
model.load_state_dict(torch.load(model_path))

jokes_output_file_path = f'generated_{MODEL_EPOCH}.jokes'

model.eval()
if os.path.exists(jokes_output_file_path):
    os.remove(jokes_output_file_path)
    
joke_num = 0
with torch.no_grad():
   
        for joke_idx in range(10):
        
            joke_finished = False

            cur_ids = torch.tensor(tokenizer.encode("JOKE:")).unsqueeze(0).to(device)

            for i in range(100):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
                if i < 3:
                    n = 1
                else:
                    n = 1
                next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n) #Randomly(from the topN probability distribution) select the next word
                cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word to the running sequence

                if next_token_id in tokenizer.encode('<|endoftext|>'):
                    joke_finished = True
                    break

            
            if joke_finished:
                
                joke_num = joke_num + 1
                
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)

                with open(jokes_output_file_path, 'a') as f:
                    f.write(f"{output_text} \n\n")


