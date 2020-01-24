import torch
gpt_version='gpt2-large' ### The version of GPT model choose between [gpt2-small, gpt2-medium, gpt2-large]
file_path='D:\Vijay Code\Personal Projects\GPT-2\Train' #### Path where all the text files are located.
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000
MAX_SEQ_LEN = 400 ### Number of sentences to be generated
models_folder = "D:\Vijay Code\Personal Projects\GPT-2\Train\models" ### Folder where the model and the result is stored.
models_folder_generate="D:\Vijay Code\Personal Projects\GPT-2\Train\models" ### If the model for txt generation is different from the
                                                                            ### trained, use this. Else, have the same path for both variables.
starting_sentence="Neuroscience" ### Starting prompt for sentence generation
generate_one_sentence=True ## if True, generates only one sentence.

##### File Structure ######
###  Train
###    |__ file1.txt
###    |__ file2.txt
###    |__ file3.txt
###    |__ file4.txt
###     ....
###    |__ models
###           |__ gpt2_0.pt
###           |__ gpt2_1.pt
###           ...
###           |__ gpt2_final.pt
###           |__ generated_sentences.txt