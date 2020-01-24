import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import json
import csv
import logging
import warnings
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule
from variables import *
from dataset import ImportDataset
logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')
print("Loading Weights..")
tokenizer = GPT2Tokenizer.from_pretrained(gpt_version)
model = GPT2LMHeadModel.from_pretrained(gpt_version)
model = model.to(device)
print("Weights Loaded..")
##### Randomly chooses the next word from the set of top n probable words #####
##### You can choose to set the value of n to be any number. Default is 5 #####
def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)
dataset = ImportDataset()
dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True)
model = model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = WarmupLinearSchedule(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps = -1)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0
tmp_sentences_tens = None

if not os.path.exists(models_folder):
    os.mkdir(models_folder)

for epoch in range(EPOCHS):
    
    print(f"EPOCH {epoch} started" + '=' * 30)
    
    for idx,sentence in enumerate(dataset_loader):
        
        #################### "Fit as many sentences into MAX_SEQ_LEN sequence as possible" logic start ####
        sentence_tens = torch.tensor(tokenizer.encode(sentence[0])).unsqueeze(0).to(device)
        # Skip sample from dataset if it is longer than MAX_SEQ_LEN
        if sentence_tens.size()[1] > MAX_SEQ_LEN:
            continue
        
        #The first sentence in the sequence
        if not torch.is_tensor(tmp_sentences_tens):
            tmp_sentences_tens = sentence_tens
            continue
        else:
            #The next sentence does not fit in so we process the sequence and leave the last sentence
            #as the start for next sequence 
            if tmp_sentences_tens.size()[1] + sentence_tens.size()[1] > MAX_SEQ_LEN:
                work_sentences_tens = tmp_sentences_tens
                tmp_sentences_tens = sentence_tens
            else:
                #Add the sentence to sequence, continue and try to add more
                tmp_sentences_tens = torch.cat([tmp_sentences_tens, sentence_tens[:,1:]], dim=1)
                continue
        ################## Sequence ready, process it trough the model ##################
            
        outputs = model(work_sentences_tens, labels=work_sentences_tens)
        loss, logits = outputs[:2]                        
        loss.backward()
        sum_loss = sum_loss + loss.detach().data
                       
        proc_seq_count = proc_seq_count + 1
        if proc_seq_count == BATCH_SIZE:
            proc_seq_count = 0    
            batch_count += 1
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad()
            model.zero_grad()
        print("\nbatch count: ",batch_count)
        if batch_count == 100:
            print(f"sum loss {sum_loss}")
            batch_count = 0
            sum_loss = 0.0
    
    # Store the model after each epoch to compare the performance of them
    torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_{epoch}.pt"))
torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_final.pt"))
print("Model Saved")
#MODEL_EPOCH = 4
