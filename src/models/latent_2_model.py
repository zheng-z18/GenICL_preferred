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
import torch.nn.functional as F
from logger_config import logger

class Latent_2_Model(nn.Module):
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
         
        # self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    
    # def batch_perplexity(self, batch):
    #     loss_fct = nn.CrossEntropyLoss(reduction='none')#reduction='mean': 默认情况下，CrossEntropyLoss 会返回平均损失（batch 中所有样本的损失取均值）
    #     outputs = self.hf_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    #     logits = outputs.logits
    #     shift_logits = logits[..., :-1, :].contiguous()
    #     shift_labels = batch['input_ids'][..., 1:].contiguous()
    #     per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    #     per_token_loss = per_token_loss.view(batch['input_ids'].size(0), -1)  # 还原为每个句子的维度
    #     sentence_lengths = (batch['attention_mask'][..., 1:] != 0).sum(dim=1)
    #     per_sentence_loss = per_token_loss.sum(dim=1) / sentence_lengths.float()
    #     perplexity = torch.exp(per_sentence_loss)
    #     assert not torch.isinf(perplexity).any(), "Perplexity contains inf values"
    #     return perplexity

    # def forward(self, batch):##batch: Dict[str, torch.Tensor]   input_ids, attention_mask, **kwargs
    #     demo_query_perplexity = self.batch_perplexity(batch)
    #     demo_query_perplexity = demo_query_perplexity.view(-1)
    #     pp_result = demo_query_perplexity[1::2] / demo_query_perplexity[0::2] ###分子分母相同，log(1)=0 是否会有影响
    #     batch_size = len(pp_result) // self.args.train_n_passages
    #     # assert batch_size == self.args.per_device_train_batch_size #batch不满的也没有填充，这里会assert
    #     pp_result_tensor = pp_result.view(batch_size, self.args.train_n_passages)
    #     epsilon = 1e-3
    #     tau = 1.0
    #     logits = -torch.log(pp_result_tensor + epsilon) / tau  # 形状：[batch_size, num_samples_per_sample]
    #     labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device) #每个样本的正样本索引为 0
    #     loss_fn = nn.CrossEntropyLoss()
    #     loss = loss_fn(logits, labels)
    #     # assert loss > 0.1, "loss < 0.1"
    #     return loss
    
    def forward(self, batch: Dict[str, torch.Tensor]):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        # 获取 batch 大小和 train_n_passages
        batch_size, train_n_passages, seq_len = input_ids.size()
        # 初始化 CrossEntropyLoss，忽略 -100 的标签
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        
        outputs = self.hf_model(input_ids=input_ids.view(-1, seq_len), attention_mask=attention_mask.view(-1, seq_len))

        logits = outputs.logits  # [batch_size * train_n_passages, seq_len, vocab_size]
        
        # 计算每个 token 的 loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().view(-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels)
        # 恢复 loss 的形状为 [batch_size * train_n_passages, seq_len]
        loss = loss.view(batch_size, train_n_passages, -1).sum(dim=-1)

        # 正样本在每一组中的索引为0，负样本为1, 2, 3
        pos_loss = loss[:, 0]  # 正样本的 loss
        neg_loss = loss[:, 1:]  # 负样本的平均 loss
        # 计算分子：正样本的 exp(logits)
        pos_exp = torch.exp(-pos_loss)
        
        # 计算分母：正样本 + 负样本的 exp(logits)
        neg_exp = torch.exp(-neg_loss).sum(dim=1)  # 对负样本取 exp 后求和
        denom = pos_exp + neg_exp
        loss = -torch.log(pos_exp / denom)  # 对每个样本计算负对数损失
        return loss.mean()
    
        #if train_n_passages == 1:
        #     loss = loss
        # else:

        #对比损失
        # pos_loss = loss[:, 0]  # [batch_size]
        # neg_loss = loss[:, 1:]  # [batch_size, 3]
        
        # # 将损失映射到相似度空间（使用损失的倒数作为相似度的代理）
        # pos_sim = 1.0 / (pos_loss + 1e-8)  # [batch_size, 1]
        # neg_sim = 1.0 / (neg_loss + 1e-8)  # [batch_size, 3]
        
        # # 连接正样本和负样本的相似度
        # sim = torch.cat([pos_sim, neg_sim], dim=1)  # [batch_size, 4]
        
        # # 应用温度缩放
        # sim = sim / self.temperature  # [batch_size, 4]
        
        # # 标签：正样本在第一位
        # target = torch.zeros(batch_size, dtype=torch.long, device=sim.device)  # [batch_size]
        
        # # 计算对比损失
        # contrastive_loss = nn.CrossEntropyLoss()(sim, target)

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
