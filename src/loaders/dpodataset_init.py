import os
import random
import torch

from copy import deepcopy
from functools import partial
from typing import Dict, List
from datasets import load_dataset, Dataset

from config import Arguments
from logger_config import logger
from utils import get_input_files
from .loader_utils import group_doc_ids, filter_invalid_examples
from data_utils import to_positive_negative_format


class DPODataset(Dataset):

    def __init__(self, args: Arguments):
        self.args = args
        self.input_files = get_input_files(self.args.train_file)
        self.negative_size = args.train_n_passages - 1
        assert self.negative_size > 0, "Negative size must be positive."

        corpus_path = os.path.join(os.path.dirname(self.input_files[0]), 'passages.jsonl.gz')
        self.corpus: Dataset = load_dataset('json', data_files=corpus_path, split='train')

        self.dataset: Dataset = load_dataset('json', data_files=self.input_files, split='train')
        self.dataset = self.dataset.filter(lambda x: x['task_name'] in args.llm_eval_tasks)
        # reduced_length = max(1, len(self.dataset) // 100)  ##############################
        # self.dataset = self.dataset.select(range(reduced_length))###########################减少数据量来debug
        logger.info(f"Filtered Dataset length: {len(self.dataset)}")

        # 预处理数据集
        with self.args.main_process_first(desc="pre-processing"):
            self.dataset = filter_invalid_examples(args, self.dataset) #
            self.dataset = self.dataset.map(
                partial(to_positive_negative_format,
                        topk_as_positive=args.topk_as_positive,
                        bottomk_as_negative=args.bottomk_as_negative),
                desc='to_positive_negative_format',
                remove_columns=['doc_ids', 'doc_scores']
            )

        # 可选：限制训练样本数量
        if self.args.max_train_samples is not None:
            self.dataset = self.dataset.select(range(self.args.max_train_samples))
        
        # 记录一个样本
        if len(self.dataset) > 0:
            sample_index = random.randint(0, len(self.dataset) - 1)
            logger.info(f"Sample {sample_index} of the training set: {self.dataset[sample_index]}.")
        else:
            logger.warning("The dataset is empty after preprocessing.")

        # 应用转换函数，添加 'prompt', 'chosen', 'rejected' 字段
        self.dataset = self.dataset.map(
            self._transform_func_dpo,
            batched=True,
            remove_columns=self.dataset.column_names,
            desc='Transforming to prompt-chosen-rejected format'
        )

        super().__init__(self.dataset.data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _transform_func_dpo(self, examples: Dict[str, List]) -> Dict[str, List]:
        # 不使用 Trainer，因此将 current_epoch 设置为 0
        current_epoch = 0  # 如果需要，可以通过 self.args 设置

        examples = deepcopy(examples)
        
        
        for idx in range(len(examples['query_id'])): ## 确保每个查询有足够的负样本
            while len(examples['negatives'][idx]['doc_id']) < self.negative_size:
                random_doc_id = str(random.randint(0, len(self.corpus) - 1))
                examples['negatives'][idx]['doc_id'].append(random_doc_id)
                examples['negatives'][idx]['score'].append(-100)

        input_doc_ids = group_doc_ids(
            examples=examples,
            negative_size=self.negative_size,
            offset=current_epoch + self.args.seed  # 根据需要调整
        )
        expected_length = len(examples['query']) * self.args.train_n_passages
        assert len(input_doc_ids) == expected_length, f"Expected {expected_length} doc_ids, got {len(input_doc_ids)}."

        prompts, chosens, rejecteds = [], [], []
        for q_idx in range(len(examples['query'])):
            query = examples['query'][q_idx]
            task_name = examples['task_name'][q_idx]
            prefix_description = ''.join([f'<{task_name}-{i}>' for i in range(self.args.n_prefix_tokens)])
            current_query = prefix_description + '\n' + query
            answers, options = examples['answers'][q_idx], examples['options'][q_idx]
            if len(options) > 1:
                current_query += '\n' + options[ord(answers[0]) - ord('A')]
            else:
                current_query += '\n' + random.choice(answers)
            
            input_docs = []
            start_idx = q_idx * self.args.train_n_passages
            end_idx = (q_idx + 1) * self.args.train_n_passages
            for idx in range(start_idx, end_idx):
                doc_id = input_doc_ids[idx]
                input_docs.append(self.corpus[doc_id]['contents'].strip())

            prompts.append(current_query)
            chosens.append(input_docs[0])
            rejecteds.append(input_docs[-1])

        return {
            'prompt': prompts,
            'chosen': chosens,
            'rejected': rejecteds
        }

    def save_dataset(self, save_path: str):
        """
        将转换后的数据集保存为 JSONL 文件。
        每行将包含 'prompt', 'chosen', 和 'rejected' 字段的 JSON 对象。
        """
        self.dataset.to_json(save_path, lines=True)
        logger.info(f"Dataset successfully saved to {save_path}.")
