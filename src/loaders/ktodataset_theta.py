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
# from data_utils import to_positive_negative_format
from functools import partial
from tasks import get_possible_answers_by_task_name

def filter_invalid_examples(args: Arguments, dataset: Dataset) -> Dataset:
    def _filter_func(example: Dict) -> bool:
        if len(example['doc_ids']) <= len(args.topk_as_positive):  # Ensure more than topk docs
            return False
        if example['task_name'] in args.held_out_tasks:  # Exclude held-out tasks
            return False

        sorted_doc_scores = sorted(example['doc_scores'], reverse=True)
        if sorted_doc_scores[len(args.topk_as_positive) - 1] <= -100.:  # Topk scores must be > -100
            return False

        return True

    return dataset.filter(
        _filter_func,
        load_from_cache_file=args.world_size > 1,
    )

def _filter_out_same_label_negatives(corpus: Dataset, example: Dict) -> Dict:
    # For text classification tasks, we need to filter out negatives examples with the same label
    # multiple choice
    if len(example['options']) > 1:
        return example
    # not text classification task
    task_name = example['task_name']
    if not get_possible_answers_by_task_name(task_name):
        return example

    answer = example['answers'][0].strip()
    new_doc_ids: List[str] = []
    new_doc_scores: List[float] = []
    for idx, doc_id in enumerate(example['negatives']['doc_id']):
        cur_ex: dict = corpus[int(doc_id)]
        contents: str = cur_ex['contents']
        input_ans: str = contents.strip().split('\n')[-1]
        if cur_ex['task_name'] != task_name or input_ans != answer:#负样本与当前任务不是一个任务，或样本答案与query答案不同：合格的负样本
            new_doc_ids.append(doc_id)
            new_doc_scores.append(example['negatives']['score'][idx])
    example['negatives']['doc_id'] = new_doc_ids
    example['negatives']['score'] = new_doc_scores
    return example

def to_positive_negative_format(
    example: Dict, 
    top_indices: List[int] = [1, 2], 
    bottom_indices: List[int] = [-1, 20, 50]
) -> Dict:
    assert len(example['doc_ids']) == len(example['doc_scores'])

    sorted_indices = np.argsort(example['doc_scores'])[::-1]  # Sort from highest to lowest
    doc_ids_sorted = [example['doc_ids'][idx] for idx in sorted_indices]
    doc_scores_sorted = [example['doc_scores'][idx] for idx in sorted_indices]
    n_docs = len(doc_ids_sorted)

    pos_doc_ids = []
    pos_doc_scores = []
    for bi in top_indices:
        assert bi < n_docs, f"Top index {bi} out of range for docs count {n_docs}"
        pos_doc_ids.append(doc_ids_sorted[bi])
        pos_doc_scores.append(doc_scores_sorted[bi])

    neg_doc_ids = []
    neg_doc_scores = []
    for bi in bottom_indices:
        if bi < 0:
            idx = n_docs + bi  # Handle negative indices
        else:
            idx = bi
        assert 0 <= idx < n_docs, f"Bottom index {bi} out of range for docs count {n_docs}"
        neg_doc_ids.append(doc_ids_sorted[idx])
        neg_doc_scores.append(doc_scores_sorted[idx])

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

class KTODataset_theta(Dataset):
    def __init__(self, args: Arguments, split: str = 'train'):
        self.args = args
        self.input_files = get_input_files(self.args.train_file)

        corpus_path = os.path.join(os.path.dirname(self.input_files[0]), 'passages.jsonl.gz')
        self.corpus: Dataset = load_dataset('json', data_files=corpus_path, split='train')
        self.dataset: Dataset = load_dataset('json', data_files=self.input_files, split='train')

        # Filter dataset to include only relevant tasks
        self.dataset = self.dataset.filter(lambda x: x['task_name'] in args.llm_eval_tasks)
        logger.info(f"Dataset length: {len(self.dataset)}")

        if split == 'train':
            split_dataset = self.dataset.train_test_split(test_size=0.1, seed=self.args.seed)
            self.dataset = split_dataset['train']
            logger.info(f"Training dataset length after split: {len(self.dataset)}")
        elif split == 'validation':
            split_dataset = self.dataset.train_test_split(test_size=0.1, seed=self.args.seed)
            self.dataset = split_dataset['test']
            logger.info(f"Validation dataset length after split: {len(self.dataset)}")
        else:
            raise ValueError("split must be 'train' or 'validation'")

        with self.args.main_process_first(desc="pre-processing"):
            # Filter out invalid examples
            self.dataset = filter_invalid_examples(args, self.dataset)
            logger.info(f"Dataset length: {len(self.dataset)} after filter_invalid_examples")
            
            # Convert to positive and negative formats
            self.dataset = self.dataset.map(
                lambda ex: to_positive_negative_format(
                    ex, 
                    top_indices=args.topk_as_positive,
                    bottom_indices=args.bottomk_as_negative
                ),
                desc='to_positive_negative_format',
                remove_columns=['doc_ids', 'doc_scores']
            )

            # self.dataset = self.dataset.map(
            #     partial(_filter_out_same_label_negatives, self.corpus),#过滤掉文本分类任务相同task不同answer的
            #     load_from_cache_file=args.world_size > 1,
            #     desc='filter_out_same_label_negatives',
            # )

        # Optionally limit the number of training samples
        if self.args.max_train_samples is not None:
            self.dataset = self.dataset.select(range(self.args.max_train_samples))
        
        if len(self.dataset) > 0:
            sample_index = random.randint(0, len(self.dataset) - 1)
            logger.info(f"Sample {sample_index} of the training set: {self.dataset[sample_index]}.")
        else:
            logger.warning("The dataset is empty after preprocessing.")

        # Transform to (prompt, completion, label) format
        self.dataset = self.dataset.map(
            self._transform_func_kto_trainer,
            batched=True,  # Use batched mapping for efficiency
            remove_columns=self.dataset.column_names,
            desc='Transforming to prompt-completion-label format'
        )

        super().__init__(self.dataset.data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _transform_func_kto_trainer(self, batch_examples: Dict[str, List]) -> Dict[str, List]:
        """
        Transforms batch examples into the (prompt, completion, label) format required by KTOTrainer.
        """
        new_prompts, new_completions, new_labels = [], [], []
        all_docs_in_batch = []
        for i in range(len(batch_examples['query'])):
            all_docs_in_batch.extend(batch_examples['positives'][i]['doc_id'])
            all_docs_in_batch.extend(batch_examples['negatives'][i]['doc_id'])
        all_docs_in_batch = list(set(all_docs_in_batch))  # 去重

        for i in range(len(batch_examples['query'])):
            query = batch_examples['query'][i]
            task_name = batch_examples['task_name'][i]
            prefix_description = ''.join([
                f'<{task_name}-{idx}>' 
                for idx in range(self.args.n_prefix_tokens)
            ]) 

            # Retrieve positive and negative document IDs
            pos_doc_ids = batch_examples['positives'][i]['doc_id']
            neg_doc_ids = batch_examples['negatives'][i]['doc_id']
            if len(neg_doc_ids) > len(pos_doc_ids):
                neg_doc_ids = random.sample(neg_doc_ids, len(pos_doc_ids))
            while len(neg_doc_ids) < len(pos_doc_ids):
                random_doc_id = str(random.randint(0, len(self.corpus) - 1))
                neg_doc_ids.append(random_doc_id)

            # desired_neg_count = 5 * len(pos_doc_ids)

            # if len(neg_doc_ids) > desired_neg_count:
            #     neg_doc_ids = random.sample(neg_doc_ids, desired_neg_count)
            # else:
            #     candidate_for_inbatch = [
            #         doc_id for doc_id in all_docs_in_batch
            #         if doc_id not in pos_doc_ids  # 不要与本 query 的正样本重合
            #     ]
            #     need_more = desired_neg_count - len(neg_doc_ids)
            #     if len(candidate_for_inbatch) > need_more:
            #         sampled_inbatch = random.sample(candidate_for_inbatch, need_more) ## 随机从 candidate_for_inbatch 里选出需要的负样本数量
            #         neg_doc_ids.extend(sampled_inbatch)
            #     else:
            #         # candidate_for_inbatch 也不够时  从 self.corpus 随机补充
            #         neg_doc_ids.extend(candidate_for_inbatch)
            #         while len(neg_doc_ids) < desired_neg_count:
            #             random_doc_id = str(random.randint(0, len(self.corpus) - 1))
            #             # 避免和当前 query 的正样本重合
            #             if random_doc_id not in pos_doc_ids:
            #                 neg_doc_ids.append(random_doc_id)

            # Create (prompt, completion, label) pairs for positive samples
            for pos_id in pos_doc_ids:
                chosen_text = self.corpus[int(pos_id)]['contents'].strip()
                if self.args.inverse == True:
                    current_prompt = prefix_description + '\n' + query
                    new_prompts.append(current_prompt)
                    new_completions.append(chosen_text)
                    new_labels.append(True)
                elif self.args.inverse == False:
                    current_prompt = chosen_text + '\n' + query
                    new_prompts.append(current_prompt)
                    new_completions.append(prefix_description)
                    new_labels.append(True)
                else:
                    raise ValueError("An error occurred due to self.args.inverse")

            # Create (prompt, completion, label) pairs for negative samples
            for neg_id in neg_doc_ids:
                rejected_text = self.corpus[int(neg_id)]['contents'].strip()
                if self.args.inverse == True:
                    current_prompt = prefix_description + '\n' + query
                    new_prompts.append(current_prompt)
                    new_completions.append(rejected_text)
                    new_labels.append(False)
                elif self.args.inverse == False:
                    current_prompt = rejected_text + '\n' + query
                    new_prompts.append(current_prompt)
                    new_completions.append(prefix_description)
                    new_labels.append(False)
                else:
                    raise ValueError("An error occurred due to self.args.inverse")
        return {
            'prompt': new_prompts,
            'completion': new_completions,
            'label': new_labels
        }

    def save_dataset(self, save_path: str):
        """
        Saves the transformed dataset to a JSONL file.
        Each line contains a JSON object with 'prompt', 'completion', and 'label'.
        """
        self.dataset.to_json(save_path, lines=True)
        logger.info(f"Dataset successfully saved to {save_path}.")