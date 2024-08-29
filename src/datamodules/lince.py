from typing import Optional, Dict, Any
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import datasets as ds
from sklearn.model_selection import KFold

from config import (
    BATCH_SIZE,
    GLOBAL_SEED,
    K_CROSSFOLD_VALIDATION_SPLITS,
    LABEL2ID,
    MAX_SEQUENCE_LENGTH,
    NUM_WORKERS,
    PADDING,
    PATH_BASE_MODELS,
    PATH_CACHE_DATASET,
    PATH_LINCE_DATASET
)

class LinceDM(pl.LightningDataModule):

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        dataset_dir=PATH_LINCE_DATASET,
        task: str = 'ner',
        batch_size: int = BATCH_SIZE,
        max_seq_len: int = MAX_SEQUENCE_LENGTH,
        padding: str = PADDING,
        label2id: dict = LABEL2ID,
        num_workers: int = NUM_WORKERS,
    ) -> None:
        super().__init__()

        self.model_name_or_path = model_name
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.task = task
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.label2id = label2id
        self.num_workers = num_workers

        if self.task in ['ner', 'pos']:
            self.data_map = {
                "train": f"{self.dataset_dir}/train.json",
                "validation": f"{self.dataset_dir}/val.json"
            }

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            cache_dir=PATH_BASE_MODELS,
            use_fast=True,
        )
    
    def prepare_data(self) -> None:
        ds.load_dataset(
            'json',
            data_files=self.data_map,
            field='data',
            cache_dir=PATH_CACHE_DATASET
        )
    
    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = ds.load_dataset(
            'json',
            data_files=self.data_map,
            field="data",
            cache_dir=PATH_CACHE_DATASET
        )

        # Apply the mapping to all splits
        self.dataset = self.dataset.map(
            self._convert_to_features,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_workers
        )

        # Set the format to PyTorch tensors
        self.dataset['train'].set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'task_no'])
        self.dataset['validation'].set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'task_no'])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )

    def _convert_to_features(self, batch: Dict[str, Any], indices=None) -> Dict[str, torch.Tensor]:
        features = self.tokenizer(
            text=batch['sentence'],
            max_length=self.max_seq_len,
            padding=self.padding,
            truncation=True,
            is_split_into_words=True,
        )    

        if self.task == 'ner':
            features["labels"] = self._align_tags(features, batch['bio_tag'])
            task_no = 0  # Assuming NER task is task 0
        elif self.task == 'pos':
            features["labels"] = self._align_tags_pos(features, batch['pos_tags'])
            task_no = 1  # Assuming POS task is task 1
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        # Add task_no to features as metadata for the model to use
        features["task_no"] = torch.tensor([task_no] * len(features['input_ids']), dtype=torch.long)

        return features

    def _align_tags(self, tokenized_outs, tags):
        batch_tags = []
        for example_id in range(len(tags)):
            example_tags = []
            current_word = None
            for word_id in tokenized_outs.word_ids(example_id):
                if word_id != current_word:
                    current_word = word_id
                    tag = len(self.label2id) if word_id is None else self.label2id.get(tags[example_id][word_id], len(self.label2id))
                    example_tags.append(tag)
                elif word_id is None:
                    example_tags.append(len(self.label2id))
                else:
                    tag = self.label2id.get(tags[example_id][word_id], len(self.label2id))
                    if tag % 2 == 1:
                        tag += 1
                    example_tags.append(tag)
            
            batch_tags.append(example_tags)

        return batch_tags

    def _align_tags_pos(self, tokenized_outs, tags):
        batch_tags = []
        for example_id in range(len(tags)):
            example_tags = []
            current_word = None
            for word_id in tokenized_outs.word_ids(example_id):
                if word_id != current_word:
                    current_word = word_id
                    tag = len(self.label2id) if word_id is None else self.label2id.get(tags[example_id][word_id], len(self.label2id))
                    example_tags.append(tag)
                elif word_id is None:
                    example_tags.append(len(self.label2id))
                else:
                    tag = self.label2id.get(tags[example_id][word_id], len(self.label2id))
                    example_tags.append(tag)
            
            batch_tags.append(example_tags)

        return batch_tags

class CrossValidationLinceDM(LinceDM):
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        k: int,
        task: str = 'ner',
        dataset_dir=PATH_LINCE_DATASET,
        batch_size: int = BATCH_SIZE,
        max_seq_len: int = MAX_SEQUENCE_LENGTH,
        padding: str = PADDING,
        label2id: dict = LABEL2ID,
        num_splits: int = K_CROSSFOLD_VALIDATION_SPLITS,
        num_workers: int = NUM_WORKERS,
        split_seed: int = GLOBAL_SEED,
    ) -> None:
        super().__init__(model_name, dataset_name, dataset_dir, task, batch_size, max_seq_len, padding, label2id, num_workers)
        self.k = k
        self.num_splits = num_splits
        self.split_seed = split_seed

    def prepare_data(self) -> None:
        ds.load_dataset(
            'json',
            data_files=f"{self.dataset_dir}/data.json",
            field='data',
            cache_dir=PATH_CACHE_DATASET
        )
    
    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = ds.load_dataset(
            'json',
            data_files=f"{self.dataset_dir}/data.json",
            field='data',
            cache_dir=PATH_CACHE_DATASET
        )

        # Random state is essential to get the same splits
        kf = KFold(n_splits=self.num_splits, shuffle=True, random_state=self.split_seed)
        splits = list(kf.split(np.zeros(len(self.dataset['train']))))
        train_idxs, val_idxs = splits[self.k]

        self.dataset = ds.DatasetDict({
            'train': self.dataset['train'].select(train_idxs),
            'validation': self.dataset['train'].select(val_idxs),
            'test': self.dataset['train'].select(val_idxs)
        })

        # Apply the mapping to all splits
        self.dataset = self.dataset.map(
            self._convert_to_features,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_workers
        )

        # Set the format to PyTorch tensors
        self.dataset['train'].set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'task_no'])
        self.dataset['validation'].set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'task_no'])
        self.dataset['test'].set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'task_no'])

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )
