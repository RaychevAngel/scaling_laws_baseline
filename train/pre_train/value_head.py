import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import os

class ValueFunction(nn.Module):
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-135M", dropout_rate: float = 0.1, 
                 load_full_model: bool = False):
        """
        Initialize the value function model.

        Args:
            model_name (str): Pretrained model name or path.
            dropout_rate (float): Dropout probability for the value head.
            load_full_model (bool): If True, loads the entire model from checkpoint instead of 
                                   initializing base model and value head separately.
        """
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.value_head = nn.Linear(self.base_model.config.hidden_size, 1)
        nn.init.kaiming_normal_(self.value_head.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.value_head.bias)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            
        Returns:
            torch.Tensor: Predicted value (regression output).
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]
        
        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1) - 1
        else:
            seq_lens = torch.tensor([input_ids.shape[-1]-1] * input_ids.shape[0])
            
        pooled = last_hidden[torch.arange(last_hidden.size(0)), seq_lens]
        return self.value_head(pooled).squeeze(-1)
            
    def prepare_input(self, conversations):
        """
        Convert conversation format to tokenized input for the model
        
        Args:
            conversations: List of conversations in the format
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        
        Returns:
            Tokenized input ready for the model with input_ids and attention_mask
        """
        formatted_texts = [conv for conv in conversations]

        # Tokenize with the properly configured tokenizer
        tokenized_inputs = self.tokenizer.apply_chat_template(
            formatted_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        # Ensure we return a dictionary with input_ids and attention_mask
        if isinstance(tokenized_inputs, torch.Tensor):
            input_ids = tokenized_inputs
            attention_mask = torch.ones_like(input_ids)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        elif hasattr(tokenized_inputs, 'input_ids'):
            input_ids = tokenized_inputs.input_ids
            if hasattr(tokenized_inputs, 'attention_mask'):
                attention_mask = tokenized_inputs.attention_mask
            else:
                attention_mask = torch.ones_like(input_ids)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        if 'attention_mask' not in tokenized_inputs:
            tokenized_inputs['attention_mask'] = torch.ones_like(tokenized_inputs['input_ids'])
        
        return tokenized_inputs
    
    def predict(self, conversations: List[Dict[str, str]]) -> torch.Tensor:
        """
        Make value predictions for a list of conversations
        
        Args:
            conversations: List containing dictionaries with role and content keys
            
        Returns:
            Value predictions (0-1 probability)
        """
        # Prepare inputs
        inputs = self.prepare_input([conversations])
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            value = self.forward(**inputs)
            value = torch.sigmoid(value)  # Apply sigmoid to get probability
            
        return value
