from src.modules.base_model import BaseModel
import torch
import torch.nn as nn
import pytorch_lightning as pl
from TorchCRF import CRF
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
        self.crf = CRF(
                num_tags=n_labels + 1,
                batch_first=True
            )
    
    def forward(self, x, labels, attention_mask):
        x = self.linear(x)
        loss = -self.crf(x, labels, attention_mask.bool())
        
        path = self.crf.decode(x)
        path = torch.Tensor(path).long()
        
        return loss, path

class SequenceMultiTaskModel(pl.LightningModule):
    def __init__(
        self,
        label2ids, # list of dicts depicting labels to be assigned. Each dict represents a task
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
        #self.baseModel.freeze()
        
        self.log_vars = []

        # Architecture: base model -> BiLSTM -> CRF
        self.bi_lstm = nn.LSTM(
            input_size=self.baseModel.config.hidden_size,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )
        
        self.shared_linear0 = nn.Linear(512, 256)
        self.shared_linear1 = nn.Linear(256, 32)
        #self.shared_linear2 = nn.Linear(128, 64)
        
        self.linear = nn.Sequential(
            self.shared_linear0, 
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            self.shared_linear1,
            nn.LayerNorm(32),
            nn.GELU(),
        )

        self.special_tag_ids = []
        for x in self.label2ids:
            self.special_tag_ids.append(len(x))

        self.task_heads = []

        for i in range(len(self.label2ids)):
            task_id = i

            self.task_heads.append(
                TaskHead( len(self.label2ids[i]) )
            )
            
            self.log_vars.append(nn.Parameter(torch.zeros(1)))

            self.add_module(f"Task {task_names[task_id]} TaskHead", self.task_heads[task_id])
            self.register_parameter(name=f"Loss param {task_names[task_id]}", param=self.log_vars[task_id])

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, task_no = batch
        
        linear_outs = self(self.baseModel(input_ids, attention_mask))
        
        outlist = [ [[], [], []] for x in range(len(self.label2ids)) ]        
        for x in range(len(linear_outs)):
            task_id = task_no[x]
            outlist[task_id][0].append(linear_outs[x])
            outlist[task_id][1].append(attention_mask[x])
            outlist[task_id][2].append(labels[x])
        
        loss = 0
        for task_id in range(len(outlist)):
            if len(outlist[task_id][0]) == 0:
                continue
            
            outlist[task_id] = [ torch.stack(y).to(device) for y in outlist[task_id] ]
            emissions, attention_mask, labels = outlist[task_id]
            outlist[task_id] = self.task_heads[task_id](emissions, labels, attention_mask)
            task_loss, task_path = outlist[task_id]
            loss += torch.exp(-self.log_vars[task_id]) * task_loss + self.log_vars[task_id]
            
            metrics = self._compute_metrics(task_path, labels, f'{self.task_names[task_id]} train', task_id)
            self.log_dict(metrics, on_step=False, on_epoch=True)

        self.log("loss/train", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, task_no = batch
        
        linear_outs = self(self.baseModel(input_ids, attention_mask))
        
        outlist = [ [[], [], []] for x in range(len(self.label2ids)) ]        
        for x in range(len(linear_outs)):
            task_id = task_no[x]
            outlist[task_id][0].append(linear_outs[x])
            outlist[task_id][1].append(attention_mask[x])
            outlist[task_id][2].append(labels[x])
        
        loss = 0
        for task_id in range(len(outlist)):
            if len(outlist[task_id][0]) == 0:
                continue
            
            outlist[task_id] = [ torch.stack(y).to(device) for y in outlist[task_id] ]
            emissions, attention_mask, labels = outlist[task_id]
            outlist[task_id] = self.task_heads[task_id](emissions, labels, attention_mask)
            task_loss, task_path = outlist[task_id]
            loss += task_loss
            
            metrics = self._compute_metrics(task_path, labels, f'{self.task_names[task_id]} val', task_id)
            self.log_dict(metrics, on_step=False, on_epoch=True)

        self.log("loss/val", loss, on_step=False, on_epoch=True)
        return loss

    def forward(self, base_model_outs):
        base_outs = base_model_outs.last_hidden_state
        lstm_outs, _ = self.bi_lstm(base_outs)
        linear_outs = self.linear(lstm_outs)

        return linear_outs
    
    def _compute_metrics(self, preds: torch.Tensor, labels: torch.Tensor, mode: str, task_id: int):

        preds = torch.reshape(preds, (-1, 1))
        preds = preds.type_as(labels)

        labels = torch.reshape(labels, (-1, 1))
        
        metrics = {}
        # metrics[f"acc/{mode}"] = accuracy(preds, labels, num_classes=len(self.label2id) + 1, ignore_index=self.special_tag_id)
        metrics[f"prec/{mode}"] = precision(preds, labels, num_classes=len(self.label2ids[task_id]) + 1, ignore_index=self.special_tag_ids[task_id], average="macro")
        metrics[f"rec/{mode}"] = recall(preds, labels, num_classes=len(self.label2ids[task_id]) + 1, ignore_index=self.special_tag_ids[task_id], average="macro")
        metrics[f"f1/{mode}"] = f1_score(preds, labels, num_classes=len(self.label2ids[task_id]) + 1, ignore_index=self.special_tag_ids[task_id], average="macro")

        return metrics
    
    def configure_optimizers(self):
        
        parameters = [
        {
            'params': self.baseModel.parameters()
        },
        {
            'params': self.bi_lstm.parameters(),
            'lr': 1e-5
        },
        {
            'params': self.linear.parameters(),
            'lr': 1e-6
        }
        ]
        
        for log_var in self.log_vars:
            parameters.append(
            {
                'params': log_var
            }
            )
        
        for name, module in self.named_modules():
            if 'Task NER TaskHead' == name:
                parameters.append(
                {
                    'params': module.parameters(),
                    'lr': 2e-6
                }
                )
            
            elif 'Task LID TaskHead' == name:
                parameters.append(
                {
                    'params': module.parameters(),
                    'lr': 5e-8
                }
                )
    
        optimizer = torch.optim.AdamW(
            params=parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        return optimizer
