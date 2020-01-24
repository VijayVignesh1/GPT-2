import csv
from variables import *
from torch.utils.data import Dataset
import os
import glob, re
print(file_path)
class ImportDataset(Dataset):
    def __init__(self, dataset_name=None, dataset_path = file_path):
        super().__init__()
        #dataset_path = os.path.join(dataset_path, dataset_name)
        self.dataset_list = []
        self.end_of_text_token = "<|endoftext|>"
        for i in glob.glob(dataset_path+"/*.txt"):
            read=open(i,'r',encoding="utf8",errors="ignore")
            lines=read.readlines()
            lines="".join(lines)
            lines_clean=re.sub('[^\nA-Za-z0-9 ]+', '', lines)
            string=f"{lines_clean}+{self.end_of_text_token}"
            self.dataset_list.append(string)
    def __len__(self):
        return len(self.dataset_list)
    def __getitem__(self, item):
        return self.dataset_list[item]