import random
import torch

from copy import deepcopy
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerFast, Trainer

from config import Arguments
from logger_config import logger
from utils import get_input_files

class SftDataset(torch.utils.data.Dataset):
    def __init__(self, input_dataset, args: Arguments):
        self.args = args
        self.dataset = input_dataset

        # self.dataset = load_dataset('json', data_files=self.input_files, split='train')
        # logger.info(f"Original Dataset length: {len(self.dataset)}")

        # self.dataset = self.dataset.filter(lambda x: x['task_name'] in args.llm_eval_tasks[0].split())
        # logger.info(f"Filtered Dataset length: {len(self.dataset)}")

        # if not args.llm_eval_tasks or args.llm_eval_tasks[0] == 'all':
        #     args.llm_eval_tasks = sorted(self.dataset.unique('task_name'))
        #     logger.info('Train all {} tasks'.format(len(args.llm_eval_tasks)))

        # if self.args.max_train_samples is not None:#限制训练样本的数量,，参数定的None，不使用
        #     self.dataset = self.dataset.select(range(self.args.max_train_samples))
        # # Log a few random samples from the training set:
        # for index in random.sample(range(len(self.dataset)), 1):
        #     logger.info(f"Sample {index} of the training set: {self.dataset[index]}.")

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
            
            task_name = examples['task_name'][idx]
            special_tokens = ''.join([f'<{task_name}-{i}>' for i in range(self.args.n_prefix_tokens)])

            if self.args.channel == False:
                current_query = special_tokens + '\n' + current_query
                input_texts.append(current_query)
                output_texts.append(real_answers)   
            else:
                current_input = special_tokens + '\n' + real_answers
                current_output = current_query
                input_texts.append(current_input)
                output_texts.append(current_output)
            
        return {'input_texts': input_texts, 'output_texts': output_texts}
    
class SftDataLoader:
    def __init__(self, sft_dataset: Dataset, args: Arguments):
        self.args = args
        self.sft_dataset = sft_dataset
        self.train_dataset = self._get_transformed_datasets()

    def set_trainer(self, trainer: Trainer):
        if self.train_dataset is not None:
            self.train_dataset.trainer = trainer

    def _get_transformed_datasets(self):
        train_dataset = None
        if self.args.train_file is not None:
            train_dataset = SftDataset(
                args=self.args,
                input_dataset=self.sft_dataset,
            )
        return train_dataset
