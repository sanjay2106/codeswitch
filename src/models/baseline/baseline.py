from typing import Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from torchmetrics.functional import precision, recall, f1_score
from TorchCRF import CRF
from transformers.optimization import AdamW

from src.modules.base_model import BaseModel
from src.modules.mtl_loss import MultiTaskLossWrapper

from config import (
    LABEL2ID,
    LEARNING_RATE,
    WARM_RESTARTS,
    WEIGHT_DECAY,
    DROPOUT_RATE,
    MAX_SEQUENCE_LENGTH,
    PADDING
)

class BaseLine(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        max_seq_len: int = MAX_SEQUENCE_LENGTH,
        padding: str = PADDING,
        label2id: dict = LABEL2ID,
        pos_label2id: dict = LABEL2ID,  # Added for POS tagging
        learning_rate: float = LEARNING_RATE,
        ner_learning_rate: float = LEARNING_RATE,
        pos_learning_rate: float = LEARNING_RATE,  # Added for POS tagging
        warm_restart_epochs: int = WARM_RESTARTS,
        weight_decay: float = WEIGHT_DECAY,
        ner_wd: float = WEIGHT_DECAY,
        pos_wd: float = WEIGHT_DECAY,  # Added for POS tagging
        dropout_rate: float = DROPOUT_RATE,
        freeze: bool = False
    ) -> None:

        super().__init__()
        self.save_hyperparameters()

        self.pos_pad_token_label = len(self.hparams.pos_label2id)
        self.ner_pad_token_label = len(self.hparams.label2id)

        # Shared params
        self.base_model = BaseModel(self.hparams.model_name)

        # Freeze pre-trained model
        if self.hparams.freeze:
            self.base_model.freeze()

        self.bi_lstm = nn.LSTM(
            input_size=self.base_model.model.config.hidden_size,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )

        self.shared_net = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU()
        )
        
        # NER Task params
        self.ner_net = nn.Sequential(
            nn.Linear(32, len(self.hparams.label2id) + 1),
            nn.LayerNorm(len(self.hparams.label2id) + 1),
        )

        self.ner_crf = CRF(
            num_tags=len(self.hparams.label2id) + 1,
            batch_first=True
        )

        # POS Task params
        self.pos_net = nn.Sequential(
            nn.Linear(32, len(self.hparams.pos_label2id) + 1),
            nn.LayerNorm(len(self.hparams.pos_label2id) + 1)
        )

        self.pos_crf = CRF(
            num_tags=len(self.hparams.pos_label2id) + 1,
            batch_first=True
        )

        self.weighted_loss = MultiTaskLossWrapper(num_tasks=2)  # POS and NER: two tasks

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        base_model_outs = self.base_model(input_ids, attention_mask)
        base_outs = base_model_outs.last_hidden_state
        lstm_outs, _ = self.bi_lstm(base_outs)
        shared_net_outs = self.shared_net(lstm_outs)

        # NER
        ner_net_outs = self.ner_net(shared_net_outs)

        # POS
        pos_net_outs = self.pos_net(shared_net_outs)

        return ner_net_outs, pos_net_outs

    def _shared_step(self, batch, mode: str):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        ner_labels = batch['ner_labels']
        pos_labels = batch['pos_labels']

        ner_emissions, pos_emissions = self(input_ids, attention_mask)

        ner_loss = -self.ner_crf(ner_emissions, ner_labels, attention_mask.bool())
        pos_loss = -self.pos_crf(pos_emissions, pos_labels, attention_mask.bool())

        ner_path = self.ner_crf.decode(ner_emissions)
        ner_path = torch.tensor(ner_path, device=self.device).long()

        pos_path = self.pos_crf.decode(pos_emissions)
        pos_path = torch.tensor(pos_path, device=self.device).long()

        loss = self.weighted_loss(ner_loss, pos_loss)

        ner_metrics = self._compute_metrics(ner_path, ner_labels, mode, "ner")
        pos_metrics = self._compute_metrics(pos_path, pos_labels, mode, "pos")

        self.log(f"loss/{mode}", loss, on_step=(mode == "train"), on_epoch=True)
        self.log(f"loss-ner/{mode}", ner_loss, on_step=(mode == "train"), on_epoch=True)
        self.log(f"loss-pos/{mode}", pos_loss, on_step=(mode == "train"), on_epoch=True)

        self.log_dict(ner_metrics, on_step=(mode == "train"), on_epoch=True)
        self.log_dict(pos_metrics, on_step=(mode == "train"), on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.bi_lstm.named_parameters() if not any(nd in n for nd in no_decay)],
            },
            {
                'params': [p for n, p in self.shared_net.named_parameters() if not any(nd in n for nd in no_decay)],
            },
            {
                'params': [p for n, p in self.ner_net.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': self.hparams.ner_learning_rate,
                'weight_decay': self.hparams.ner_wd
            },
            {
                'params': [p for n, p in self.pos_net.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': self.hparams.pos_learning_rate,
                'weight_decay': self.hparams.pos_wd
            },
            {
                'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        if not self.hparams.freeze:
            optimizer_grouped_parameters.append({
                'params': [p for n, p in self.base_model.named_parameters() if not any(nd in n for nd in no_decay)],
            })

        optimizer = AdamW(
            params=optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=self.hparams.warm_restart_epochs,
        )

        return [optimizer], [lr_scheduler]

    def _compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor, mode: str, task: str):
        preds = preds.reshape(-1, 1)
        preds = preds.type_as(targets)  # Ensure preds tensor is on the same device as targets

        targets = targets.reshape(-1, 1)

        metrics = {}

        if task == "ner":
            metrics[f"prec/{mode}-{task}"] = precision(
                preds, targets,
                average="macro",
                num_classes=len(self.hparams.label2id) + 1,
                ignore_index=self.ner_pad_token_label
            )

            metrics[f"rec/{mode}-{task}"] = recall(
                preds, targets,
                average="macro",
                num_classes=len(self.hparams.label2id) + 1,
                ignore_index=self.ner_pad_token_label
            )

            metrics[f"f1/{mode}-{task}"] = f1_score(
                preds, targets,
                average="macro",
                num_classes=len(self.hparams.label2id) + 1,
                ignore_index=self.ner_pad_token_label
            )

        elif task == "pos":
            metrics[f"prec/{mode}-{task}"] = precision(
                preds, targets,
                average="macro",
                num_classes=len(self.hparams.pos_label2id) + 1,
                ignore_index=self.pos_pad_token_label
            )
            metrics[f"rec/{mode}-{task}"] = recall(
                preds, targets,
                average="macro",
                num_classes=len(self.hparams.pos_label2id) + 1,
                ignore_index=self.pos_pad_token_label
            )

            metrics[f"f1/{mode}-{task}"] = f1_score(
                preds, targets,
                average="macro",
                num_classes=len(self.hparams.pos_label2id) + 1,
                ignore_index=self.pos_pad_token_label
            )

        return metrics

