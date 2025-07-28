import torch
import torch.nn as nn

from typing import Optional, Dict
from transformers import (
    PreTrainedModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from config import Arguments


class MultiThetaModel(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, args: Arguments):
        super().__init__()
        self.hf_model = hf_model
        self.args = args

        self.n_tokens = args.n_cluster * args.n_prefix_tokens * len(args.llm_eval_tasks) # args.task_num
        self.orig_vocab_size = self.hf_model.get_input_embeddings().weight.size(0)
        self.hf_model.resize_token_embeddings(self.orig_vocab_size + self.n_tokens) #扩大词汇表，添加在原有词汇表的后面
        self.new_vocab_size = self.hf_model.get_input_embeddings().weight.size(0)
        assert self.new_vocab_size == self.n_tokens + self.orig_vocab_size
        self.hf_model.get_input_embeddings().weight.data[-self.n_tokens:] = \
                self.hf_model.get_input_embeddings().weight.data[:self.n_tokens] #老的现有的词汇表的前self.n_tokens个标记的嵌入
        self.hf_model.tie_weights() #将input embedding，output embedding 参数共享

        for param in self.hf_model.parameters():
            param.requires_grad = False
        
        embedding = self.hf_model.get_input_embeddings()##只训练added token的embedding参数
        embedding.weight.requires_grad = True
        def freeze_original_embeddings(grad):
            if grad.shape[0] == self.new_vocab_size:
                grad[:self.orig_vocab_size] = 0
            return grad
        embedding.weight.register_hook(freeze_original_embeddings)

        for name, param in self.hf_model.named_parameters():
            if param.requires_grad:
                print('parameter for updating')
                print(name, param.requires_grad)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, batch: Dict[str, torch.Tensor]):
        outputs = self.hf_model(**batch, return_dict=True, use_cache=False)
        labels = batch['labels']
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        per_token_loss = self.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) #计算每个token的损失
        per_sequence_loss = per_token_loss.view(batch['input_ids'].size(0), -1).sum(dim=1)
        num_valid_labels = torch.sum(labels != -100, dim=1).float()
        outputs.loss = (per_sequence_loss / num_valid_labels)
        return outputs

    def gradient_checkpointing_enable(self):
        self.hf_model.gradient_checkpointing_enable()

    @classmethod
    def from_pretrained(cls, all_args: Arguments, *args, **kwargs):
        hf_model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)
        return cls(hf_model, all_args)

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)


class RerankerForInference(nn.Module):
    def __init__(self, hf_model: Optional[PreTrainedModel] = None):
        super().__init__()
        self.hf_model = hf_model
        self.hf_model.eval()

    @torch.no_grad()
    def forward(self, batch):
        return self.hf_model(**batch)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        hf_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)
        return cls(hf_model)
