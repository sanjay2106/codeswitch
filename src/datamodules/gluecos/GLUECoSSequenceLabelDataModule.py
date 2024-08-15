import pytorch_lightning as pl
from src.datamodules.gluecos.task import Task
from config import PADDING

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _align_tags(tokenized_outs, tags, label2id):
            batch_tags = []
            for example_id in range(len(tokenized_outs['input_ids'])):
                example_tags = []
                currentWord = None
                for word_id in tokenized_outs.word_ids(example_id):

                    if (word_id != currentWord):
                        currentWord = word_id 
                        tag = len(label2id) if word_id is None else label2id[tags[example_id][word_id]]
                        example_tags.append(tag)
                    
                    elif word_id is None:
                        example_tags.append(len(label2id))
                    
                    else:
                        tag = label2id[tags[example_id][word_id]]

                        if tag % 2 == 1:
                            tag += 1
                        
                        example_tags.append(tag)
                
                batch_tags.append(example_tags)
            
            return torch.tensor(batch_tags)

class TaskDataset(Dataset):
      def __init__(self, tasksData):
        self.datapoints = []

        task_no = 0
        for taskData in tasksData:
          taskData[1] = _align_tags(taskData[0], taskData[1], taskData[2])
          for i in range(len(taskData[0]['input_ids'])):
            datapoint = [
                torch.tensor(taskData[0]['input_ids'][i]),
                torch.tensor(taskData[0]['attention_mask'][i]),
                torch.tensor(taskData[1][i]),
                torch.tensor(task_no)
            ]

            self.datapoints.append(datapoint)
          task_no += 1
        
        random.shuffle(self.datapoints)
        
      def __len__(self):
        return len(self.datapoints)

      def __getitem__(self, i):
        return self.datapoints[i][0], self.datapoints[i][1], self.datapoints[i][2], self.datapoints[i][3]

class GLUECoSSequenceLabelDataModule(pl.LightningDataModule):
    def __init__(self, tasks, max_seq_len, base_model, batch_size, num_workers):
        super().__init__()
        
        self.tasks = tasks
        self.max_seq_len = max_seq_len
        self.base_model = base_model
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def prepare_data(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
    
    def setup(self, stage):
        self.train_data = []
        for task in self.tasks:
            self.train_data.append(self._read_gluecos_(task.train_path))
            self.train_data[-1] = self._mtokenize_(self.train_data[-1], task.label2id)
        self.training_dataset = TaskDataset(self.train_data)
    
        self.val_data = []
        for task in self.tasks:
            self.val_data.append(self._read_gluecos_(task.val_path))
            self.val_data[-1] = self._mtokenize_(self.val_data[-1], task.label2id)
        self.validation_dataset = TaskDataset(self.val_data)
            
    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def _mtokenize_(self, datapoints, word2id):
      return [
          self.tokenizer(
                text=[ x[0] for x in datapoints ], 
                max_length=self.max_seq_len,
                padding=PADDING, 
                truncation=True,
                is_split_into_words=True,
            ),
          [ x[1] for x in datapoints ],
          word2id
          ]
    
    def _read_gluecos_(self, file_path):
        toret = []
        with open(file_path, encoding='utf-8') as f:
            datapoint = [[], []]
            
            for line in f.readlines():
                line = line.strip()
                
                if len(line) == 0 and len(datapoint[0]) != 0:
                    toret.append(datapoint)
                    datapoint = [[], []]
                else:
                    try:
                        split_line = line.split('\t')
                        datapoint[0].append(split_line[0])
                        datapoint[1].append(split_line[1])
                    except:
                        datapoint = [[], []] #just reset the datapoint.
            
            toret.append(datapoint)
        return toret
