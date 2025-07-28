import torch
import torch.nn as nn

from typing import Optional, Dict
from transformers import (
    PreTrainedModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from transformers.modeling_outputs import SequenceClassifierOutput, CausalLMOutputWithCrossAttentions

from config import Arguments


class DPO_Model(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, args: Arguments):
        super().__init__()
        self.hf_model = hf_model
        self.args = args

        # self.hf_model.get_input_embeddings().weight.detach().requires_grad = False   #comparison

        self.n_tokens = args.n_prefix_tokens * len(args.llm_eval_tasks) # args.task_num
        self.orig_vocab_size = self.hf_model.get_input_embeddings().weight.size(0)
        self.hf_model.resize_token_embeddings(self.orig_vocab_size + self.n_tokens) #扩大词汇表，添加在原有词汇表的后面
        self.new_vocab_size = self.hf_model.get_input_embeddings().weight.size(0)
        assert self.new_vocab_size == self.n_tokens + self.orig_vocab_size
        self.hf_model.get_input_embeddings().weight.data[-self.n_tokens:] = \
                self.hf_model.get_input_embeddings().weight.data[:self.n_tokens] #老的现有的词汇表的前self.n_tokens个标记的嵌入
        self.hf_model.tie_weights() #将input embedding，output embedding 参数共享

        # self.hf_model.get_input_embeddings().weight.data[self.orig_vocab_size:].requires_grad = True  #comparison

        for param in self.hf_model.parameters():
            param.requires_grad = False
        
        embedding = self.hf_model.get_input_embeddings()##只训练added token的embedding参数
        embedding.weight.requires_grad = True
        def freeze_original_embeddings(grad):
            if grad.shape[0] == self.new_vocab_size:
                grad[:self.orig_vocab_size] = 0
            return grad
        embedding.weight.register_hook(freeze_original_embeddings)
         
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
 
    def gradient_checkpointing_enable(self):
        self.hf_model.gradient_checkpointing_enable()

    @classmethod
    def from_pretrained(cls, all_args: Arguments, *args, **kwargs):
        hf_model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)
        return cls(hf_model, all_args)

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)
