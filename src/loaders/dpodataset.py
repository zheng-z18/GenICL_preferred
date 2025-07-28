import os
import random
import torch
import numpy as np

from copy import deepcopy
from typing import Dict, List
from datasets import load_dataset, Dataset

from config import Arguments
from logger_config import logger
from utils import get_input_files
from data_utils import to_positive_negative_format

def filter_invalid_examples(args: Arguments, dataset: Dataset) -> Dataset:
    def _filter_func(example: Dict) -> bool:
        if len(example['doc_ids']) <= len(args.topk_as_positive):#一个query检索出的docs要大于3个
            return False
        if example['task_name'] in args.held_out_tasks: #['qnli', 'piqa', 'yelp']
            return False

        sorted_doc_scores = sorted(example['doc_scores'], reverse=True)
        if sorted_doc_scores[len(args.topk_as_positive) - 1] <= -100.: #topk_as_positive 的文档得分必须大于 -100,相同task
            return False

        return True

    return dataset.filter(
        _filter_func,
        load_from_cache_file=args.world_size > 1,
    )


def to_positive_negative_format(
    example: Dict, 
    top_indices: List[int] = [1, 2], 
    bottom_indices: List[int] = [-1, 20, 50]
) -> Dict:
    assert len(example['doc_ids']) == len(example['doc_scores'])

    sorted_indices = np.argsort(example['doc_scores'])[::-1]  # 最高分 -> 最低分
    doc_ids_sorted = [example['doc_ids'][idx] for idx in sorted_indices]
    doc_scores_sorted = [example['doc_scores'][idx] for idx in sorted_indices]
    n_docs = len(doc_ids_sorted)

    pos_doc_ids = []
    pos_doc_scores = []
    for bi in top_indices:
        assert bi <= n_docs
        pos_doc_ids.append(doc_ids_sorted[bi])
        pos_doc_scores.append(doc_scores_sorted[bi])

    neg_doc_ids = []
    neg_doc_scores = []
    for bi in bottom_indices:
        assert bi <= n_docs
        neg_doc_ids.append(doc_ids_sorted[bi])
        neg_doc_scores.append(doc_scores_sorted[bi])

    return {
        'positives': {
            'doc_id': pos_doc_ids,
            'score': pos_doc_scores,
        },
        'negatives': {
            'doc_id': neg_doc_ids,
            'score': neg_doc_scores,
        },
    }


class DPODataset(Dataset):
    def __init__(self, args: Arguments):
        self.args = args
        self.input_files = get_input_files(self.args.train_file)

        corpus_path = os.path.join(os.path.dirname(self.input_files[0]), 'passages.jsonl.gz')
        self.corpus: Dataset = load_dataset('json', data_files=corpus_path, split='train')
        self.dataset: Dataset = load_dataset('json', data_files=self.input_files, split='train')

        self.dataset = self.dataset.filter(lambda x: x['task_name'] in args.llm_eval_tasks)
        logger.info(f"Dataset length: {len(self.dataset)}")

        with self.args.main_process_first(desc="pre-processing"):
            self.dataset = filter_invalid_examples(args, self.dataset)
            logger.info(f"Dataset length: {len(self.dataset)} after filter_invalid_examples")
            self.dataset = self.dataset.map(
                lambda ex: to_positive_negative_format(
                    ex, 
                    top_indices=args.topk_as_positive,
                    bottom_indices=args.bottomk_as_negative
                ),
                desc='to_positive_negative_format',
                remove_columns=['doc_ids', 'doc_scores']
            )

        if self.args.max_train_samples is not None: # 可选：限制训练样本数量
            self.dataset = self.dataset.select(range(self.args.max_train_samples))
        
        if len(self.dataset) > 0:# 打印一个样本
            sample_index = random.randint(0, len(self.dataset) - 1)
            logger.info(f"Sample {sample_index} of the training set: {self.dataset[sample_index]}.")
        else:
            logger.warning("The dataset is empty after preprocessing.")

        # 将数据转换成 (prompt, chosen, rejected) 格式
        self.dataset = self.dataset.map(
            self._transform_func_dpo,
            batched=True,            # 注意这里要使用 batched=True 才能一次返回多条
            remove_columns=self.dataset.column_names,
            desc='Transforming to prompt-chosen-rejected format'
        )

        super().__init__(self.dataset.data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _transform_func_dpo(self, batch_examples: Dict[str, List]) -> Dict[str, List]:
        """
        对每条样本，组合正样本列表和负样本列表，构造多对 (prompt, chosen, rejected)。
        假设:
          - positives['doc_id'] 里有若干个正样本
          - negatives['doc_id'] 里有若干个负样本
        则最终生成的数量是 `len(positives) * len(negatives)`.
        """
        batch_size = len(batch_examples['query'])
        new_prompts, new_chosens, new_rejecteds = [], [], []

        for i in range(batch_size):
            query = batch_examples['query'][i]
            task_name = batch_examples['task_name'][i]
            prefix_description = ''.join([
                f'<{task_name}-{idx}>' 
                for idx in range(self.args.n_prefix_tokens)
            ])
            current_query = prefix_description + '\n' + query

            # 如果是选择题
            answers = batch_examples['answers'][i]
            options = batch_examples['options'][i]
            if len(options) > 1:
                # 如果是 A/B/C/D 多选，answers[0] 形如 'A'/'B' 等
                correct_idx = ord(answers[0]) - ord('A')  
                if 0 <= correct_idx < len(options):
                    current_query += '\n' + options[correct_idx]
                else:
                    # 如果答案越界，就随便选一个，或其他处理
                    current_query += '\n' + random.choice(options)
            else:
                # 如果不是选择题，直接加一个答案
                current_query += '\n' + random.choice(answers)

            # 取得正负样本 doc_id
            pos_doc_ids = batch_examples['positives'][i]['doc_id']
            neg_doc_ids = batch_examples['negatives'][i]['doc_id']

            # 两两组合，生成多条 (prompt, chosen, rejected)
            for pos_id in pos_doc_ids:
                chosen_text = self.corpus[int(pos_id)]['contents'].strip()
                for neg_id in neg_doc_ids:
                    rejected_text = self.corpus[int(neg_id)]['contents'].strip()
                    new_prompts.append(current_query)
                    new_chosens.append(chosen_text)
                    new_rejecteds.append(rejected_text)

        return {
            'prompt': new_prompts,
            'chosen': new_chosens,
            'rejected': new_rejecteds
        }

    def save_dataset(self, save_path: str):
        """
        将转换后的数据集保存为 JSONL 文件。
        每行将包含 'prompt', 'chosen', 和 'rejected' 字段的 JSON 对象。
        """
        self.dataset.to_json(save_path, lines=True)
        logger.info(f"Dataset successfully saved to {save_path}.")
