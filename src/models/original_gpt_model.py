import torch
import torch.nn as nn

from typing import Optional, Dict
from transformers import (
    PreTrainedModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
# from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from config import Arguments


class OriginalGPTModel(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, args: Arguments):
        super().__init__()
        self.hf_model = hf_model
        self.args = args

        # self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, batch: Dict[str, torch.Tensor]):
        outputs: CausalLMOutputWithCrossAttentions = self.hf_model(**batch, return_dict=True, use_cache=False)
        labels = batch['labels']
        shift_logits = outputs.logits[..., :-1, :].contiguous() #左移一个位置，使得每个位置能够预测下一个位置
        shift_labels = labels[..., 1:].contiguous()
        per_token_loss = self.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) #计算每个token的损失
        per_sequence_loss = per_token_loss.view(batch['input_ids'].size(0), -1).sum(dim=1)
        num_valid_labels = torch.sum(labels != -100, dim=1).float()
        outputs.loss = (per_sequence_loss / num_valid_labels)  #.cpu().tolist()
        # avg_log_probs += (-per_sequence_loss / num_valid_labels).cpu().tolist() #平均对数损失
        return outputs
    # def forward(self, batch: Dict[str, torch.Tensor]) -> SequenceClassifierOutput:
    #     input_batch_dict = {k: v for k, v in batch.items() if k != 'labels'}#其中input_ids [16, 384]
    #     #batch包括input_ids,attention_mask,labels。input_batch_dict去除掉labels，只保留前两项
    #     outputs: SequenceClassifierOutput = self.hf_model(**input_batch_dict, return_dict=True)#[16,384,50257]
    #     # outputs.logits = outputs.logits.view(-1, self.args.train_n_passages)
    #     loss = self.cross_entropy(outputs.logits, batch['labels'])#batch['labels']
    #     outputs.loss = loss
    #     return outputs

    def gradient_checkpointing_enable(self):
        self.hf_model.gradient_checkpointing_enable()

    @classmethod
    def from_pretrained(cls, all_args: Arguments, *args, **kwargs):
        hf_model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)
        # hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
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
