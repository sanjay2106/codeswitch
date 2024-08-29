import torch
import torch.nn as nn 
import pytorch_lightning as pl 
from transformers import AutoConfig, AutoModel 
from config import PATH_BASE_MODELS

class BaseModel(pl.LightningModule):
    def __init__(
        self, 
        model_name: str,
    ) -> None:
        super().__init__()

        self.model_name = model_name

        # Load model configuration and model itself from pre-trained model
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.model_name, 
            cache_dir=PATH_BASE_MODELS, 
        )

        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name, 
            cache_dir=PATH_BASE_MODELS
        )
    
    def forward(self, input_ids, attention_mask):
        # Ensure that input_ids and attention_mask are tensors
        print(f"Type of input_ids: {type(input_ids)}")
        print(f"Type of attention_mask: {type(attention_mask)}")

        assert isinstance(input_ids, torch.Tensor), "input_ids must be a tensor"
        assert isinstance(attention_mask, torch.Tensor), "attention_mask must be a tensor"
        
        # Pass the inputs through the model and return the output
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


