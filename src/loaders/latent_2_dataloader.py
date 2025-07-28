import os.path
import random
import torch

from copy import deepcopy
from functools import partial
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset
from transformers.file_utils import PaddingStrategy
from transformers import PreTrainedTokenizerFast, Trainer

from config import Arguments
from logger_config import logger
from utils import get_input_files
from .loader_utils import group_doc_ids, filter_invalid_examples
from data_utils import to_positive_negative_format


class Latent2Dataset(torch.utils.data.Dataset):

    def __init__(self, input_files: List[str], args: Arguments,
                 tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.input_files = input_files
        self.negative_size = args.train_n_passages - 1 #8-1=7
        # assert self.negative_size > 0
        self.tokenizer = tokenizer
        corpus_path = os.path.join(os.path.dirname(self.input_files[0]), 'passages.jsonl.gz')
        self.corpus: Dataset = load_dataset('json', data_files=corpus_path, split='train')

        self.dataset: Dataset = load_dataset('json', data_files=self.input_files, split='train')
        self.dataset = self.dataset.filter(lambda x: x['task_name'] in args.llm_eval_tasks)
        logger.info(f"Filtered Dataset length: {len(self.dataset)}")

        # with self.args.main_process_first(desc="pre-processing"):#只有主进程会执行该块代码
        #     self.dataset = filter_invalid_examples(args, self.dataset)
        #     self.dataset = self.dataset.map(
        #         partial(to_positive_negative_format,
        #                 topk_as_positive=args.topk_as_positive,#指定正样本的数量 3
        #                 bottomk_as_negative=args.bottomk_as_negative),#负样本数量  16
        #         load_from_cache_file=args.world_size > 1,
        #         desc='to_positive_negative_format',
        #         remove_columns=['doc_ids', 'doc_scores']
        #     )
        with self.args.main_process_first(desc="pre-processing"):
            self.dataset = filter_invalid_examples(args, self.dataset)
            self.dataset = self.dataset.map(
                partial(to_positive_negative_format,
                        topk_as_positive=args.topk_as_positive,#指定正样本的数量 3
                        bottomk_as_negative=args.bottomk_as_negative),#负样本数量  16
                num_proc=1,
                # load_from_cache_file=args.world_size > 1,
                load_from_cache_file=True,
                desc='to_positive_negative_format',
                remove_columns=['doc_ids', 'doc_scores']
            )


        if self.args.max_train_samples is not None:
            self.dataset = self.dataset.select(range(self.args.max_train_samples))
        # Log a few random samples from the training set:
        for index in random.sample(range(len(self.dataset)), 1):
            logger.info(f"Sample {index} of the training set: {self.dataset[index]}.")

        # self.dataset.set_transform(self._transform_func_ppl)
        self.dataset.set_transform(self._transform_func_theta)
        self.trainer: Optional[Trainer] = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _transform_func_theta(self, examples: Dict[str, List]) -> Dict[str, List]:
        current_epoch = int(self.trainer.state.epoch or 0) if self.trainer is not None else 0

        examples = deepcopy(examples)
        # add some random negatives if not enough
        for idx in range(len(examples['query_id'])):
            while len(examples['negatives'][idx]['doc_id']) < self.negative_size:#负样本不足
                random_doc_id = str(random.randint(0, len(self.corpus) - 1))#从corpus中随机选doc_id
                examples['negatives'][idx]['doc_id'].append(random_doc_id)
                examples['negatives'][idx]['score'].append(-100.)

        input_doc_ids = group_doc_ids( #input_doc_ids第一个是正样本的id，后7个是负样本的id
            examples=examples,
            negative_size=self.negative_size,
            offset=current_epoch + self.args.seed##offset参数，在不同的epoch中选取不同的正负样本，增加数据多样性，防止过拟合
        )
        assert len(input_doc_ids) == len(examples['query']) * self.args.train_n_passages #每个query有train_n_passages个样本
        task_name = examples['task_name'][0]
        prefix_description = ''.join([f'<{task_name}-{i}>' for i in range(self.args.n_prefix_tokens)])

        input_queries, input_docs = [], []
        input_texts = []
        output_texts = []
        for idx, doc_id in enumerate(input_doc_ids):
            demonstration = self.corpus[doc_id]['contents'].strip() #
            input_docs.append(self.corpus[doc_id]['contents'].strip()) #
            q_idx = idx // self.args.train_n_passages
            current_query: str = demonstration + '\n' + examples['query'][q_idx]
            input_texts.append(current_query)
            output_texts.append(prefix_description)
        
        # import json
        # output_file = "output_agnews.json"
        # data_to_save = {'input_texts': input_texts, 'output_texts': output_texts, 'answer': examples['answers'][0]}
        # with open(output_file, "a") as f:
        #     f.write("\n")
        #     json.dump(data_to_save, f, ensure_ascii=False)
        return {'input_texts': [input_texts], 'output_texts': [output_texts]}

class Latent2DataLoader:

    def __init__(self, args: Arguments, tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.tokenizer = tokenizer
        # self.train_dataset, self.eval_dataset = self._get_transformed_datasets()
        self.train_dataset = self._get_transformed_datasets()

    def set_trainer(self, trainer: Trainer):
        if self.train_dataset is not None:
            self.train_dataset.trainer = trainer
        # if self.eval_dataset is not None:
        #     self.eval_dataset.trainer = trainer

    def _get_transformed_datasets(self) -> Latent2Dataset:
        train_dataset = None
        # eval_dataset = None

        if self.args.train_file is not None:
            train_input_files = get_input_files(self.args.train_file)
            logger.info("Train files: {}".format(train_input_files))
            train_dataset = Latent2Dataset( #latent_dataset
                args=self.args,
                tokenizer=self.tokenizer,
                input_files=train_input_files,
            )
            # train_dataset = latent_dataset.train_dataset
            # eval_dataset = latent_dataset.val_dataset

        # if self.args.do_train:
        #     assert train_dataset is not None, "Training requires a train dataset"
        # return train_dataset, eval_dataset
        return train_dataset



    # def _transform_func_ppl(self, examples: Dict[str, List]) -> Dict[str, List]:
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
    #         offset=current_epoch + self.args.seed##offset参数，在不同的epoch中选取不同的正负样本，增加数据多样性，防止过拟合
    #     )
    #     assert len(input_doc_ids) == len(examples['query']) * self.args.train_n_passages #每个query有train_n_passages个样本
        
    #     task_name = examples['task_name'][0]
    #     prefix_description = ''.join([f'<{task_name}-{i}>' for i in range(self.args.n_prefix_tokens)])

    #     combined_sentences: List[str] = []
    #     for idx, doc_id in enumerate(input_doc_ids):#根据input_doc_ids从corpus中获取doc内容，添加到input_docs
    #         # input_docs.append(self.corpus[doc_id]['contents'].strip())
    #         # For reward model, the left side is the query + ground truth answer
    #         q_idx = idx // self.args.train_n_passages #//整除，一直为0
    #         current_query: str = examples['query'][q_idx]

    #         single_demo_query = '{}\n\n{}'.format(self.corpus[int(doc_id)]['contents'].strip(), current_query)
    #         # demo_query_sentences.append(single_demo_query)
    #         single_demo_query_description = '{}\n\n{}'.format(prefix_description, single_demo_query)
    #         # description_demo_query_sentences.append(single_demo_query_description)
    #         combined_sentences.append(single_demo_query)
    #         combined_sentences.append(single_demo_query_description)

    #     batch_dict = self.tokenizer(combined_sentences,
    #                                 max_length=self.args.reward_max_length,
    #                                 return_token_type_ids=False,
    #                                 padding=PaddingStrategy.DO_NOT_PAD,
    #                                 truncation=True)
    #     #batch_dict中有input_ids和attention_mask
    #     packed_batch_dict = {}
    #     for k in batch_dict:# k=input_ids
    #         packed_batch_dict[k] = []
    #         assert len(examples['query']) * self.args.train_n_passages * 2 == len(batch_dict[k])
    #         for idx in range(len(examples['query'])):
    #             start = idx * self.args.train_n_passages * 2
    #             packed_batch_dict[k].append(batch_dict[k][start:(start + self.args.train_n_passages * 2)])
    #         #确保每个query对应train_n_passages个docs
    #     return packed_batch_dict

    # def _transform_func_cl(self, examples: Dict[str, List]) -> Dict[str, List]:
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
    #         offset=current_epoch + self.args.seed##offset参数，在不同的epoch中选取不同的正负样本，增加数据多样性，防止过拟合
    #     )
    #     assert len(input_doc_ids) == len(examples['query']) * self.args.train_n_passages #每个query有train_n_passages个样本
        
    #     task_name = examples['task_name'][0]
    #     prefix_description = ''.join([f'<{task_name}-{i}>' for i in range(self.args.n_prefix_tokens)])

    #     input_queries, input_docs = [], []
    #     for idx, doc_id in enumerate(input_doc_ids):
    #         input_docs.append(self.corpus[doc_id]['contents'].strip())
    #         # For reward model, the left side is the query + ground truth answer
    #         q_idx = idx // self.args.train_n_passages
    #         current_query: str = prefix_description + '\n' + examples['query'][q_idx]
    #         answers, options = examples['answers'][q_idx], examples['options'][q_idx]
    #         if len(options) > 1:
    #             current_query += '\n' + options[ord(answers[0]) - ord('A')]
    #             # logger.info('current_query: %s', current_query)
    #         else:
    #             current_query += '\n' + random.choice(answers)
    #         input_queries.append(current_query)

    #     batch_dict = self.tokenizer(input_queries,# 4个query完全一样，这是一组
    #                                 text_pair=input_docs,#四个不同的doc(x,y)，第一个为正样本
    #                                 max_length=self.args.reward_max_length,
    #                                 return_token_type_ids=False,
    #                                 padding=PaddingStrategy.DO_NOT_PAD,
    #                                 truncation=True)

    #     packed_batch_dict = {}
    #     for k in batch_dict:
    #         packed_batch_dict[k] = []
    #         assert len(examples['query']) * self.args.train_n_passages == len(batch_dict[k])
    #         for idx in range(len(examples['query'])):
    #             start = idx * self.args.train_n_passages
    #             packed_batch_dict[k].append(batch_dict[k][start:(start + self.args.train_n_passages)])

    #     return packed_batch_dict

