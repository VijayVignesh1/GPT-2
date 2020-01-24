from variables import *
from dataset import ImportDataset
import torch
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from gpt2_train import *
tokenizer = GPT2Tokenizer.from_pretrained(gpt_version)
model = GPT2LMHeadModel.from_pretrained(gpt_version)
model=model.to(device)
print("Model Loaded..")
model_path = os.path.join(models_folder, f"gpt2_final.pt")
model.load_state_dict(torch.load(model_path))

sentences_output_file_path = f'generated_sentences.txt'
sentences_output_file_path=os.path.join(models_folder_generate,sentences_output_file_path)
model.eval()
if os.path.exists(sentences_output_file_path):
    os.remove(sentences_output_file_path)
    
sentence_num = 0
print("Generating sentences..")

start="Me: "+ starting_sentence+"\n"

with torch.no_grad():
   
        for sentence_idx in range(1000):
        
            sentence_finished = False

            cur_ids = torch.tensor(tokenizer.encode(start)).unsqueeze(0).to(device)

            for i in range(100):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
                if i < 3:
                    n = 20
                else:
                    n = 3
                next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n) #Randomly(from the topN probability distribution) select the next word
                cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word to the running sequence

                if next_token_id in tokenizer.encode('<|endoftext|>'):
                    sentence_finished = True
                    break
            if sentence_finished:
                sentence_num += 1
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)
                with open(sentences_output_file_path, 'a') as f:
                    f.write(f"{output_text} \n\n")
                    if generate_one_sentence==True:
                        print(output_text)
                        break