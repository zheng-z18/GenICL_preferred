import random
import torch
import json
from copy import deepcopy
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerFast, Trainer

from config import Arguments
from logger_config import logger
from utils import get_input_files

class MultiThetaDataset(torch.utils.data.Dataset):
    def __init__(self, input_files: List[str], args: Arguments): #tokenizer: PreTrainedTokenizerFast
        self.args = args
        self.input_files = input_files

        self.dataset = load_dataset('json', data_files=self.input_files, split='train')
        logger.info(f"Original Dataset length: {len(self.dataset)}")

        self.dataset = self.dataset.filter(lambda x: x['task_name'] in args.llm_eval_tasks[0].split())
        logger.info(f"Filtered Dataset length: {len(self.dataset)}")

        with open(self.args.json_path, 'r') as f:
            cluster_to_data = json.load(f)

        query_to_cluster = {}
        for cluster, data_ids in cluster_to_data.items():
            for data_id in data_ids:
                query_to_cluster[data_id] = int(cluster)

        def add_cluster(example):
            data_id = example['query_id']
            cluster = query_to_cluster.get(data_id, None)  # 如果没有找到，返回 None
            return {'cluster': cluster}
        self.dataset = self.dataset.map(add_cluster, desc="Adding cluster information")

        logger.info(f"Dataset with cluster column has {len(self.dataset)} entries.")
        print(self.dataset.column_names)  # 查看所有列名，确保 'cluster' 已添加
        print(self.dataset[0])  # 查看第一个数据点，确认 'cluster' 值


        if not args.llm_eval_tasks or args.llm_eval_tasks[0] == 'all':
            args.llm_eval_tasks = sorted(self.dataset.unique('task_name'))
            logger.info('Train all {} tasks'.format(len(args.llm_eval_tasks)))

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
            
            task_name = examples['task_name'][idx]

            cluster = examples['cluster'][idx]
            special_tokens = ''.join([f'<{task_name}-{cluster}-{i}>' for i in range(self.args.n_prefix_tokens)])

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
    
class MultiThetaDataLoader:
    # def __init__(self, args: Arguments, tokenizer: PreTrainedTokenizerFast):
    def __init__(self, args: Arguments):
        self.args = args
        # self.tokenizer = tokenizer
        self.train_dataset = self._get_transformed_datasets()

    def set_trainer(self, trainer: Trainer):
        if self.train_dataset is not None:
            self.train_dataset.trainer = trainer

    def _get_transformed_datasets(self):
        train_dataset = None

        if self.args.train_file is not None:
            train_input_files = get_input_files(self.args.train_file)#用来处理有多个输入文件
            logger.info("Train files: {}".format(train_input_files))
            train_dataset = MultiThetaDataset(
                args=self.args,
                # tokenizer=self.tokenizer,
                input_files=train_input_files,
            )

        if self.args.do_train:
            assert train_dataset is not None, "Training requires a train dataset"

        return train_dataset
