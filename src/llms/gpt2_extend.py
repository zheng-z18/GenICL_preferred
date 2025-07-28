import torch
from tqdm import tqdm
import numpy as np

from contextlib import nullcontext
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from typing import List
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.generation.utils import GreedySearchDecoderOnlyOutput
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from utils import move_to_device
from logger_config import logger
from config import Arguments
from llms.base_llm import BaseLLM
from collators.gpt2_collator import ScoreCollator, PerplexityCollator, DecodeCollator


class GPT2_extend(BaseLLM):

    def __init__(self, args: Arguments, accelerator, model_name_or_path: str = 'gpt2-xl', **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.args = args
        self.accelerator = accelerator
        ###############################################修改tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
            additional_special_tokens=[f'<{task}-{i}>' #为不同任务创建独特的标记，帮助模型区分不同的任务
                for task in args.llm_eval_tasks
                for i in range(args.n_prefix_tokens)])#为每个任务生成n_prefix_tokens
        prefix_token_ids = {}
        for i, task in enumerate(args.llm_eval_tasks):
            prefix_token_ids[task] = \
                self.tokenizer.additional_special_tokens_ids[
                    i*args.n_prefix_tokens: (i+1)*args.n_prefix_tokens]
        ###################
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.truncation_side = 'left'

        self.batch_size_per_device = args.llm_batch_size_per_device

        dtype = torch.float16 if args.fp16 else torch.float32
        self.model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
        ###############################################修改model
        self.n_tokens = args.n_prefix_tokens * len(args.llm_eval_tasks)
        self.orig_vocab_size = self.model.get_input_embeddings().weight.size(0)
        self.model.resize_token_embeddings(self.orig_vocab_size + self.n_tokens) #扩大词汇表，添加在原有词汇表的后面
        self.new_vocab_size = self.model.get_input_embeddings().weight.size(0)
        assert self.new_vocab_size == self.n_tokens + self.orig_vocab_size
        
        # self.model.set_input_embeddings(torch.load(args.prefix_embed_file, map_location=torch.device('cuda:0'))) #从train载入weight
        # prefix_embed_weights = torch.load(args.prefix_embed_file, map_location=torch.device('cuda:0'))
        prefix_embed_weights = torch.load(args.prefix_embed_file, map_location='cpu')
        self.model.get_input_embeddings().load_state_dict(prefix_embed_weights)
        
        self.model.tie_weights() #将input embedding，output embedding 参数共享
        ################
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.eval()

        self.model = self.accelerator.prepare(self.model)

    @torch.no_grad()
    def batch_score(
            self, input_texts: List[str], output_texts: List[str],
            delimiter: str = '\n', **kwargs
    ) -> List[float]:
        assert len(input_texts) == len(output_texts), '{} != {}'.format(len(input_texts), len(output_texts))
        assert not all(output in ['A', 'B', 'C', 'D'] for output in output_texts), 'output_texts should not be letters'

        collator = ScoreCollator(
            tokenizer=self.tokenizer,
            max_length=self.args.llm_max_input_length,
            pad_to_multiple_of=8,    #填充到8的倍数
            delimiter=delimiter,   #分隔符
        )

        dataset = Dataset.from_dict({ #使用 input 和 output 构建数据集
            'input_texts': input_texts,
            'output_texts': output_texts
        })
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=2,
            collate_fn=collator,
            pin_memory=True
        )
        
        data_loader = self.accelerator.prepare(data_loader)
        
        avg_log_probs: List[float] = [] 
        loss_fct = CrossEntropyLoss(reduction='none')
        for batch_dict in tqdm(data_loader, desc='batch score', mininterval=10, disable=len(dataset) < 1024):
            # Hack: remove token_type_ids for llama model
            if 'llama' in self.model_name_or_path and 'token_type_ids' in batch_dict:
                del batch_dict['token_type_ids']

            # batch_dict = move_to_device(batch_dict, device=self.model.device) #batch_dict包括input_ids，attention_mask，labels
            # with torch.amp.autocast('cuda') if self.args.fp16 else nullcontext():
            outputs: CausalLMOutputWithCrossAttentions = self.model(
                **batch_dict, return_dict=True, use_cache=False
            )# outputs 包含logits和loss

            labels = batch_dict['labels']
            shift_logits = outputs.logits[..., :-1, :].contiguous() #左移一个位置，使得每个位置能够预测下一个位置
            shift_labels = labels[..., 1:].contiguous()
            per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) #计算每个token的损失
            per_sequence_loss = per_token_loss.view(batch_dict['input_ids'].size(0), -1).sum(dim=1)
            num_valid_labels = torch.sum(labels != -100, dim=1).float()
            avg_log_probs += (-per_sequence_loss / num_valid_labels).cpu().tolist()

        return avg_log_probs

    
    def batch_decode(self, input_texts: List[str], prefix_trie=None, **kwargs) -> List[str]:
        collator = DecodeCollator(
            tokenizer=self.tokenizer,
            max_length=self.args.llm_max_input_length,
            pad_to_multiple_of=8
        )
        dataset: Dataset = Dataset.from_dict({'input_texts': input_texts})
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=2,
            collate_fn=collator,
            pin_memory=True
        )

        decoded_texts: List[str] = []
        eos_token_id: int = self.tokenizer.encode('\n')[-1]
        for batch_dict in tqdm.tqdm(data_loader, mininterval=10, desc='batch decode'):
            # Hack: remove token_type_ids for llama model
            if 'llama' in self.model_name_or_path and 'token_type_ids' in batch_dict:
                del batch_dict['token_type_ids']

            batch_dict = move_to_device(batch_dict, device=self.model.device)
            input_len: int = batch_dict['input_ids'].shape[1]
            
            newline_id = self.tokenizer.encode('\n')[-1]
            def _prefix_allowed_tokens_fn(_, generated_ids):
                returned = prefix_trie.get(generated_ids.tolist()[input_len:])
                if len(returned) == 0:
                    returned = [newline_id]
                return returned

            with torch.amp.autocast('cuda') if self.args.fp16 else nullcontext():
                outputs: GreedySearchDecoderOnlyOutput = self.model.generate(
                    **batch_dict,
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=self.args.llm_max_decode_length,
                    begin_suppress_tokens=[eos_token_id],
                    eos_token_id=eos_token_id, #指定结束符号的token id
                    prefix_allowed_tokens_fn=_prefix_allowed_tokens_fn if prefix_trie else None,
                    return_dict_in_generate=True,
                    output_scores=False,#不返回生成token的分数
                )
                generated_token_ids = outputs.sequences[:, input_len:]#从input_len开始截取生成的token ID
                logger.debug('generated_token_ids: {}'.format(generated_token_ids.tolist()))

                if outputs.scores is not None:
                    transition_scores = self.model.compute_transition_scores(
                        outputs.sequences, outputs.scores, normalize_logits=True
                    )
                    for tok, score in zip(generated_token_ids[0].cpu(), transition_scores[0].cpu()):
                        if tok in self.tokenizer.all_special_ids:
                            continue
                        # | token | token string | logits | probability
                        logger.info(f"| {tok:5d} | {self.tokenizer.decode(tok):8s} | {score.numpy():.4f} "
                                    f"| {np.exp(score.numpy()):.2%}")

            decoded_texts += self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)#根据token id生成文本

        return decoded_texts

    

    @torch.no_grad()
    def batch_score_init(
            self, input_texts: List[str], output_texts: List[str],
            delimiter: str = '\n', **kwargs
    ) -> List[float]:
        assert len(input_texts) == len(output_texts), '{} != {}'.format(len(input_texts), len(output_texts))
        assert not all(output in ['A', 'B', 'C', 'D'] for output in output_texts), 'output_texts should not be letters'

        collator = ScoreCollator(
            tokenizer=self.tokenizer,
            max_length=self.args.llm_max_input_length,
            pad_to_multiple_of=8,    #填充到8的倍数
            delimiter=delimiter,   #分隔符
        )

        dataset = Dataset.from_dict({ #使用 input 和 output 构建数据集
            'input_texts': input_texts,
            'output_texts': output_texts
        })
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=2,
            collate_fn=collator,
            pin_memory=True
        )

        avg_log_probs: List[float] = [] 
        loss_fct = CrossEntropyLoss(reduction='none')
        for batch_dict in tqdm(data_loader, desc='batch score', mininterval=10, disable=len(dataset) < 1024):
            # Hack: remove token_type_ids for llama model
            if 'llama' in self.model_name_or_path and 'token_type_ids' in batch_dict:
                del batch_dict['token_type_ids'] #llama模型不需要token_type_ids

            batch_dict = move_to_device(batch_dict, device=self.model.device) #batch_dict包括input_ids，attention_mask，labels
            with torch.amp.autocast('cuda') if self.args.fp16 else nullcontext():
                outputs: CausalLMOutputWithCrossAttentions = self.model(
                    **batch_dict, return_dict=True, use_cache=False
                )# outputs 包含logits和loss

                labels = batch_dict['labels']
                shift_logits = outputs.logits[..., :-1, :].contiguous() #左移一个位置，使得每个位置能够预测下一个位置
                shift_labels = labels[..., 1:].contiguous()
                per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) #计算每个token的损失
                per_sequence_loss = per_token_loss.view(batch_dict['input_ids'].size(0), -1).sum(dim=1)
                num_valid_labels = torch.sum(labels != -100, dim=1).float()
                avg_log_probs += (-per_sequence_loss / num_valid_labels).cpu().tolist()

                # avg_log_probs += (-per_sequence_loss / num_valid_labels).cpu().tolist() #平均对数损失

                # logger.debug('num_valid_labels: {}, loss: {}, per_token_loss: {}, avg_per_token_loss: {}'.format(
                #     num_valid_labels, outputs.loss, per_token_loss,
                #     per_token_loss.sum() / torch.sum(labels != -100).float())
                # )

        return avg_log_probs #avg_log_probs是32000个，input_texts和output_texts都是32000个
