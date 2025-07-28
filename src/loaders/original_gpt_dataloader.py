import random
import torch

from copy import deepcopy
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerFast, Trainer

from config import Arguments
from logger_config import logger
from utils import get_input_files

class OriginalGPTDataset(torch.utils.data.Dataset):
    def __init__(self, input_files: List[str], args: Arguments,
                 tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.input_files = input_files
        self.tokenizer = tokenizer

        self.dataset: Dataset = load_dataset('json', data_files=self.input_files, split='train')
        if self.args.max_train_samples is not None:#限制训练样本的数量,，参数定的None，不使用
            self.dataset = self.dataset.select(range(self.args.max_train_samples))
        # Log a few random samples from the training set:
        for index in random.sample(range(len(self.dataset)), 1):
            logger.info(f"Sample {index} of the training set: {self.dataset[index]}.")

        self.dataset.set_transform(self._transform_func)
        self.trainer: Optional[Trainer] = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _transform_func(self, examples: Dict[str, List]) -> Dict[str, List]:
        examples = deepcopy(examples)
        assert len(examples['query']) == 1, f"Expected 'query' length to be 1, but got {len(examples['query'])}"
        input_texts = []
        output_texts = []
        for idx in range(len(examples['query'])):
            current_query: str = examples['query'][idx]
            answers, options = examples['answers'][idx], examples['options'][idx]
            if len(options) > 1:
                real_answers = options[ord(answers[0]) - ord('A')]
            else:
                real_answers = random.choice(answers)
            input_texts.append(current_query)
            output_texts.append(real_answers)
        return {'input_texts': input_texts, 'output_texts': output_texts}
    
class OriginalGPTDataLoader:
    def __init__(self, args: Arguments, tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.tokenizer = tokenizer
        self.train_dataset = self._get_transformed_datasets()

    def set_trainer(self, trainer: Trainer):
        if self.train_dataset is not None:
            self.train_dataset.trainer = trainer

    def _get_transformed_datasets(self):
        train_dataset = None

        if self.args.train_file is not None:
            train_input_files = get_input_files(self.args.train_file)#用来处理有多个输入文件
            logger.info("Train files: {}".format(train_input_files))
            train_dataset = OriginalGPTDataset(
                args=self.args,
                tokenizer=self.tokenizer,
                input_files=train_input_files,
            )

        if self.args.do_train:
            assert train_dataset is not None, "Training requires a train dataset"

        return train_dataset



    # def _transform_func_init(self, examples: Dict[str, List]) -> Dict[str, List]:
    #     current_epoch = int(self.trainer.state.epoch or 0) if self.trainer is not None else 0

    #     examples = deepcopy(examples)
    #     # add some random negatives if not enough
    #     for idx in range(len(examples['query_id'])):
    #         while len(examples['negatives'][idx]['doc_id']) < self.negative_size:#负样本不足
    #             random_doc_id = str(random.randint(0, len(self.corpus) - 1))#从corpus中随机选doc_id
    #             examples['negatives'][idx]['doc_id'].append(random_doc_id)
    #             examples['negatives'][idx]['score'].append(-100.)

    #     input_doc_ids = group_doc_ids( #input_doc_ids第一个是正样本的id，后7个是负样本的id
    #         examples=examples,
    #         negative_size=self.negative_size,
    #         offset=current_epoch + self.args.seed
    #     )
    #     assert len(input_doc_ids) == len(examples['query']) * self.args.train_n_passages #每个query有train_n_passages个样本

    #     input_queries, input_docs = [], []
    #     for idx, doc_id in enumerate(input_doc_ids):#根据input_doc_ids从corpus中获取doc内容，添加到input_docs
    #         input_docs.append(self.corpus[doc_id]['contents'].strip())
    #         # For reward model, the left side is the query + ground truth answer
    #         q_idx = idx // self.args.train_n_passages #//整除，一直为0
    #         current_query: str = examples['query'][q_idx]
    #         answers, options = examples['answers'][q_idx], examples['options'][q_idx]
    #         if len(options) > 1:
    #             current_query += '\n' + options[ord(answers[0]) - ord('A')] #多选题 将选项添加到query后
    #             # logger.info('current_query: %s', current_query)
    #         else:
    #             current_query += '\n' + random.choice(answers)#从答案中随机选一个加到query后
    #         input_queries.append(current_query)#因为q_idx一直为0，这几个完全一样

    #     batch_dict = self.tokenizer(input_queries, #对input_queries和input_docs编码
    #                                 text_pair=input_docs,
    #                                 max_length=self.args.reward_max_length,
    #                                 return_token_type_ids=False,
    #                                 padding=PaddingStrategy.DO_NOT_PAD,
    #                                 truncation=True)
    #     #batch_dict中有input_ids和attention_mask
    #     packed_batch_dict = {}
    #     for k in batch_dict:# k=input_ids
    #         packed_batch_dict[k] = []
    #         assert len(examples['query']) * self.args.train_n_passages == len(batch_dict[k])
    #         for idx in range(len(examples['query'])):
    #             start = idx * self.args.train_n_passages
    #             packed_batch_dict[k].append(batch_dict[k][start:(start + self.args.train_n_passages)])
    #         #确保每个query对应train_n_passages个docs
    #     return packed_batch_dict #与batch_dict相比基本不变

    # def _transform_func_1(self, examples: Dict[str, List]) -> Dict[str, List]:
    #     examples = deepcopy(examples)
    #     assert len(examples['query']) == 1, f"Expected 'query' length to be 1, but got {len(examples['query'])}"
    #     input_queries = []
    #     for idx in range(len(examples['query'])):
    #         current_query: str = examples['query'][idx]
    #         answers, options = examples['answers'][idx], examples['options'][idx]
    #         if len(options) > 1:
    #             current_query += '\n' + options[ord(answers[0]) - ord('A')] #多选题 将选项添加到query后
    #             # logger.info('current_query: %s', current_query)
    #         else:
    #             assert len(answers) == 1, f"Answer is not only one"
    #             add_answer = random.choice(answers) #应该只有一个答案
    #             current_query += '\n' + add_answer  #从答案中随机选一个加到query后
    #         input_queries.append(current_query)
    #     batch_dict = self.tokenizer(input_queries, #对input_queries和input_docs编码
    #                                 max_length=self.args.reward_max_length,#????
    #                                 return_token_type_ids=False,
    #                                 padding=PaddingStrategy.DO_NOT_PAD,
    #                                 truncation=True)
    #     #batch_dict中有input_ids和attention_mask
    #     packed_batch_dict = {}
    #     for k in batch_dict:# k=input_ids
    #         packed_batch_dict[k] = []
    #         for idx in range(len(examples['query'])):
    #             packed_batch_dict[k].append(batch_dict[k])
    #     return packed_batch_dict

