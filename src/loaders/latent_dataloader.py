import random
import torch

from copy import deepcopy
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerFast, Trainer

from config import Arguments
from logger_config import logger
from utils import get_input_files

class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, input_files: List[str], args: Arguments): #tokenizer: PreTrainedTokenizerFast
        self.args = args
        self.input_files = input_files
        # self.tokenizer = tokenizer

        self.dataset = load_dataset('json', data_files=self.input_files, split='train')
        logger.info(f"Original Dataset length: {len(self.dataset)}")

        self.dataset = self.dataset.filter(lambda x: x['task_name'] in args.llm_eval_tasks[0].split())
        logger.info(f"Filtered Dataset length: {len(self.dataset)}")

        if not args.llm_eval_tasks or args.llm_eval_tasks[0] == 'all':
            args.llm_eval_tasks = sorted(self.dataset.unique('task_name'))
            logger.info('Train all {} tasks'.format(len(args.llm_eval_tasks)))

        if self.args.max_train_samples is not None:#限制训练样本的数量,，参数定的None，不使用
            self.dataset = self.dataset.select(range(self.args.max_train_samples))
        # Log a few random samples from the training set:
        for index in random.sample(range(len(self.dataset)), 1):
            logger.info(f"Sample {index} of the training set: {self.dataset[index]}.")

        # train_val_split = self.dataset.train_test_split(test_size=0.2, seed=args.seed)
        # self.train_dataset = train_val_split['train']
        # self.val_dataset = train_val_split['test']
        # self.train_dataset.set_transform(self._transform_func)
        # self.val_dataset.set_transform(self._transform_func)
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

            current_query = special_tokens + '\n' + current_query
            input_texts.append(current_query)
            output_texts.append(real_answers)   
            
        return {'input_texts': input_texts, 'output_texts': output_texts}
    
class LatentDataLoader:
    # def __init__(self, args: Arguments, tokenizer: PreTrainedTokenizerFast):
    def __init__(self, args: Arguments):
        self.args = args
        # self.tokenizer = tokenizer
        self.train_dataset = self._get_transformed_datasets()
        # self.train_dataset, self.eval_dataset = self._get_transformed_datasets()

    def set_trainer(self, trainer: Trainer):
        if self.train_dataset is not None:
            self.train_dataset.trainer = trainer
        # if self.eval_dataset is not None:
        #     self.eval_dataset.trainer = trainer

    def _get_transformed_datasets(self):
        train_dataset = None
        # eval_dataset = None

        if self.args.train_file is not None:
            train_input_files = get_input_files(self.args.train_file)#用来处理有多个输入文件
            logger.info("Train files: {}".format(train_input_files))
            train_dataset = LatentDataset(
                args=self.args,
                # tokenizer=self.tokenizer,
                input_files=train_input_files,
            )
            # latent_dataset = LatentDataset(
            #     args=self.args,
            #     input_files=train_input_files,
            # )
            # train_dataset = latent_dataset.train_dataset
            # eval_dataset = latent_dataset.val_dataset

        if self.args.do_train:
            assert train_dataset is not None, "Training requires a train dataset"

        return train_dataset
        # return train_dataset, eval_dataset