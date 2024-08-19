from src.modules.base_model import BaseModel
from torchcrf import CRF
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, precision, recall, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TaskHead(nn.Module):
    def __init__(self, n_labels):
        super(TaskHead, self).__init__()
        self.n_labels = n_labels
        
        self.linear = nn.Sequential(
            nn.Linear(32, n_labels + 1),
            nn.LayerNorm(n_labels + 1)
        )
        self.crf = CRF(num_tags=n_labels + 1, batch_first=True)
    
    def forward(self, x, labels, attention_mask):
        x = self.linear(x)
        loss = -self.crf(x, labels, attention_mask.bool())
        
        path = self.crf.decode(x)
        path = torch.tensor(path).long().to(device)
        
        return loss, path

class SequenceMultiTaskModel(pl.LightningModule):
    def __init__(
        self,
        label2ids, 
        task_names,
        model_name_or_path,
        padding,
        learning_rate,
        weight_decay,
    ) -> None:
        super(SequenceMultiTaskModel, self).__init__()

        self.model_name_or_path = model_name_or_path
        self.label2ids = label2ids
        self.padding = padding 
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.task_names = task_names
        
        self.save_hyperparameters()

        self.baseModel = BaseModel(self.model_name_or_path)
        
        self.log_vars = []

        self.bi_lstm = nn.LSTM(
            input_size=self.baseModel.config.hidden_size,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )
        
        self.shared_linear0 = nn.Linear(512, 256)
        self.shared_linear1 = nn.Linear(256, 32)
        
        self.linear = nn.Sequential(
            self.shared_linear0, 
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            self.shared_linear1,
            nn.LayerNorm(32),
            nn.GELU(),
        )

        self.special_tag_ids = [len(x) for x in self.label2ids]

        self.task_heads = nn.ModuleList([
            TaskHead(len(self.label2ids[i])) for i in range(len(self.label2ids))
        ])
        
        for i, task_name in enumerate(task_names):
            self.log_vars.append(nn.Parameter(torch.zeros(1)))
            self.add_module(f"Task {task_name} TaskHead", self.task_heads[i])
            self.register_parameter(name=f"Loss param {task_name}", param=self.log_vars[i])

    def configure_optimizers(self):
        parameters = [
            {'params': self.baseModel.parameters()},
            {'params': self.bi_lstm.parameters(), 'lr': 1e-5},
            {'params': self.linear.parameters(), 'lr': 1e-6},
        ]
        
        for log_var in self.log_vars:
            parameters.append({'params': log_var})
        
        for i, task_name in enumerate(self.task_names):
            task_head = self.task_heads[i]
            if task_name == 'NER':
                parameters.append({'params': task_head.parameters(), 'lr': 2e-6})
            elif task_name == 'LID':
                parameters.append({'params': task_head.parameters(), 'lr': 5e-8})
        
        optimizer = torch.optim.AdamW(
            params=parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        return optimizer
