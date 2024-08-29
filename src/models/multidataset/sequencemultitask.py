import pytorch_lightning as pl
import torch
import torch.nn as nn
from TorchCRF import CRF  # Corrected the import to `torchcrf`
from torchmetrics.functional import precision as tm_precision, recall, f1_score, accuracy
import logging
from src.modules.base_model import BaseModel
import torchmetrics.functional as tmf

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

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
    
    def forward(self, x, labels=None, attention_mask=None):
        assert isinstance(x, torch.Tensor), f"x is not a Tensor, but {type(x)}"
        if labels is not None:
            assert isinstance(labels, torch.Tensor), f"labels are not a Tensor, but {type(labels)}"
        if attention_mask is not None:
            assert isinstance(attention_mask, torch.Tensor), f"attention_mask is not a Tensor, but {type(attention_mask)}"
        
        logging.debug(f"TaskHead forward | x shape: {x.shape}, labels shape: {labels.shape if labels is not None else 'N/A'}, attention_mask shape: {attention_mask.shape if attention_mask is not None else 'N/A'}")
        
        emissions = self.linear(x)
        print(f"Emissions shape: {emissions.shape}")
        if labels is not None and attention_mask is not None:
            loss = -self.crf(emissions, labels, attention_mask.bool())
            path = self.crf.decode(emissions)
            path = torch.Tensor(path).long()
            print(f"Loss: {loss.item()}, Path: {path}")
            return loss, path
        else:
            path = self.crf.decode(emissions)
            path = torch.Tensor(path).long()
            print(f"Path: {path}")
            return path

class SequenceMultiTaskModel(pl.LightningModule):
    def __init__(
        self,
        label2ids,  # list of dicts depicting labels to be assigned. Each dict represents a task
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
        self.task_names = task_names  # Expecting this to be a list
        
        self.save_hyperparameters()

        self.baseModel = BaseModel(self.model_name_or_path)
        
        self.log_vars = []

        # Architecture: base model -> BiLSTM -> CRF
        self.bi_lstm = nn.LSTM(
            input_size=self.baseModel.config.hidden_size,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )
        
        self.shared_linear0 = nn.Linear(512, 256)  # Adjust input size to match BiLSTM output
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
            TaskHead(len(labels)) for labels in self.label2ids
        ])
        
        self.log_vars = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in self.label2ids
        ])

    def forward(self, input_ids, attention_mask, labels=None, task_no=None):
        base_model_outs = self.baseModel(input_ids=input_ids, attention_mask=attention_mask)
        lstm_outs, _ = self.bi_lstm(base_model_outs.last_hidden_state)
        linear_outs = self.linear(lstm_outs)
        
        print(f"Base model outputs shape: {base_model_outs.last_hidden_state.shape}")
        print(f"LSTM outputs shape: {lstm_outs.shape}")
        print(f"Linear outputs shape: {linear_outs.shape}")
        
        if labels is not None and task_no is not None:
            loss, task_path = self.task_heads[task_no](linear_outs, labels, attention_mask)
            return loss, task_path
        else:
            task_path = self.task_heads[task_no](linear_outs)
            return task_path

    def training_step(self, batch, batch_idx):
        if len(batch) != 4:
            raise ValueError(f"Expected batch with 4 elements, but got {len(batch)} elements.")

        input_ids, attention_mask, labels, task_no = batch
        
        logging.debug(f"Training step | input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}, labels shape: {labels.shape}, task_no: {task_no}")

        loss, task_path = self.forward(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels, task_no=task_no)
        
        outlist = [ [[], [], []] for _ in range(len(self.label2ids)) ]        
        for x in range(len(task_path)):
            task_id = task_no[x]
            outlist[task_id][0].append(task_path[x])
            outlist[task_id][1].append(attention_mask[x])
            outlist[task_id][2].append(labels[x])
        
        total_loss = 0
        for task_id in range(len(outlist)):
            if len(outlist[task_id][0]) == 0:
                continue
            
            outlist[task_id] = [torch.stack(y).to(device) for y in outlist[task_id]]
            emissions, attention_mask, labels = outlist[task_id]
            print(f"Task ID {task_id}: Emissions shape: {emissions.shape}, Attention Mask shape: {attention_mask.shape}, Labels shape: {labels.shape}")
            
            task_loss, task_path = self.task_heads[task_id](emissions, labels, attention_mask)
            total_loss += torch.exp(-self.log_vars[task_id]) * task_loss + self.log_vars[task_id]
            
            metrics = self._compute_metrics(task_path, labels, f'{self.task_names[task_id]} train', task_id)
            self.log_dict(metrics, on_step=False, on_epoch=True)

        self.log("loss/train", total_loss, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Handle 'task_no' only if it's present in the batch
        task_no = batch.get('task_no', None)

        logging.debug(f"Validation step | input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}, labels shape: {labels.shape}")

        base_model_outs = self.baseModel(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        
        lstm_outs, _ = self.bi_lstm(base_model_outs.last_hidden_state)
        
        linear_outs = self.linear(lstm_outs)
        print(f"Base model outputs shape: {base_model_outs.last_hidden_state.shape}")
        print(f"LSTM outputs shape: {lstm_outs.shape}")
        print(f"Linear outputs shape: {linear_outs.shape}")

        if task_no is not None:
            val_loss = 0
            for i in range(len(task_no)):
                current_task_no = task_no[i]
                emissions = linear_outs[i:i+1]
                mask = attention_mask[i:i+1]
                label = labels[i:i+1]
                
                task_loss, task_path = self.task_heads[current_task_no](emissions, label, mask)
                val_loss += torch.exp(-self.log_vars[current_task_no]) * task_loss + self.log_vars[current_task_no]
                
                metrics = self._compute_metrics(task_path, label, f'{self.task_names[current_task_no]} val', current_task_no)
                self.log_dict(metrics, on_step=False, on_epoch=True)

            self.log("loss/val", val_loss, on_step=False, on_epoch=True)
            return val_loss
        else:
            return {"val_loss": torch.tensor(0.0)}

    def _compute_metrics(self, preds: torch.Tensor, labels: torch.Tensor, mode: str, task_id: int):
        print(f"Computing metrics for task_id: {task_id}, mode: {mode}")
        
        preds = torch.reshape(preds, (-1, len(self.label2ids[task_id]) + 1))
        labels = torch.reshape(labels, (-1, len(self.label2ids[task_id]) + 1))
        
        num_classes = len(self.label2ids[task_id]) + 1  # Total number of classes including the special tag

        metrics = {}
        
        print(f"Task type: multilabel, num_classes: {num_classes}")

        # Ensure num_classes is an integer and not None
        if num_classes is None:
            raise ValueError("num_classes must be an integer and cannot be None.")
        
        metrics[f"prec/{mode}"] = tmf.precision(
            preds=preds, 
            target=labels, 
            num_classes=num_classes, 
            ignore_index=None, 
            average="macro",
            task='multilabel'
        )
        
        metrics[f"rec/{mode}"] = tmf.recall(
            preds=preds, 
            target=labels, 
            num_classes=num_classes, 
            ignore_index=None, 
            average="macro",
            task='multilabel'
        )
        
        metrics[f"f1/{mode}"] = tmf.f1_score(
            preds=preds, 
            target=labels, 
            num_classes=num_classes, 
            ignore_index=None, 
            average="macro",
            task='multilabel'
        )
        
        metrics[f"acc/{mode}"] = tmf.accuracy(
            preds=preds, 
            target=labels, 
            num_classes=num_classes, 
            ignore_index=None, 
            average="macro",
            task='multilabel'
        )

        return metrics

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # Optionally, define a learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        # Return optimizer and scheduler as a dictionary
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
