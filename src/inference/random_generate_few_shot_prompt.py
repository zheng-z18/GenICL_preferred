import os
import sys
import random  # Added for random sampling
sys.path.insert(0, 'src/')
from tqdm import tqdm

from datasets import Dataset, load_dataset, DownloadMode, concatenate_datasets
from typing import List, Tuple
from transformers import HfArgumentParser

from config import Arguments
from logger_config import logger
from utils import save_dataset
from evaluation import BaseEval
from model_utils import build_eval_model
from inference.inference_utils import get_prompt_save_path
from llms import BaseLLM
from model_utils import build_extend_llm

# import debugpy
# try:
#     debugpy.listen(("localhost", 9609))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]


def main():
    out_path: str = get_prompt_save_path(args=args)
    if os.path.exists(out_path):
        logger.info('Prompt file {} exists. Skip.'.format(out_path))
        return

    corpus: Dataset = load_dataset(
        'json', data_files='{}/passages.jsonl.gz'.format(args.data_dir), split='train')
    # columns: query_id / query / answers / task_name
    eval_dataset: Dataset = load_dataset(
        'json', data_files='{}/{}.jsonl.gz'.format(args.data_dir, args.llm_eval_split), split='train',download_mode='force_redownload')
    args.llm_eval_tasks = sorted(args.llm_eval_tasks)
    # if not args.llm_eval_tasks or args.llm_eval_tasks[0] == 'all':
    #     real_llm_eval_tasks = sorted(eval_dataset.unique('task_name')) #30个任务
    #     args.llm_eval_tasks = real_llm_eval_tasks
        # args.llm_eval_tasks = sorted([task for task in real_llm_eval_tasks if task not in {'mnli_m', 'mnli_mm', 'wsc273'}] + ['mnli'])
        # args.llm_eval_tasks = sorted(eval_dataset.unique('task_name'))  #最初的任务选择
        # logger.info('Eval all {} tasks'.format(len(real_llm_eval_tasks)))

    # model: BaseEval = build_eval_model(args=args, corpus=corpus)

    task_ds_list: List[Dataset] = []
    for task_name in args.llm_eval_tasks:  # args.llm_eval_tasks,,real_llm_eval_tasks
        task_ds: Dataset = eval_dataset.filter(lambda x: x['task_name'] == task_name)
        if task_name in ['mnli_m', 'mnli_mm']:
            task_name = 'mnli'
        if task_name == 'wsc273':
            task_name = 'wsc'

        if len(task_ds) > args.max_test_samples:
            logger.info('Task: {}, random sample {}/{} for evaluation'.format(
                task_name, args.max_test_samples, len(task_ds))
            )
            task_ds = task_ds.shuffle(seed=args.seed).select(range(args.max_test_samples))
        logger.info('Task: {}, {} samples for evaluation'.format(task_name, len(task_ds)))

        if args.llm_k_shot <= 0:
            task_ds = task_ds.add_column('input_prompt', ['' for _ in range(len(task_ds))])
            task_ds_list.append(task_ds)
            continue

        # Randomly select topk_doc_ids from the corpus
        num_corpus = len(corpus)
        k = args.llm_k_shot

        topk_doc_ids = [random.sample(range(num_corpus), k) for _ in range(len(task_ds))]
        
        def get_prompt_by_doc_ids(doc_ids: List[int]) -> str:
            prompts = []
            for doc_id in doc_ids:
                corpus_entry = corpus[doc_id]
                if int(corpus_entry['id']) != doc_id:  # Assuming 'id' matches the index
                    raise ValueError(f"doc_id {doc_id} does not match corpus entry id {corpus_entry['id']}")
                prompts.append(corpus_entry['contents'])
            return '\n\n'.join(prompts)

        input_prompts: List[str] = [get_prompt_by_doc_ids(doc_ids) for doc_ids in topk_doc_ids]
        task_ds = task_ds.add_column('input_prompt', input_prompts)
        task_ds = task_ds.add_column('topk_doc_ids', topk_doc_ids)
        # Removed 'topk_scores' since it's no longer needed
        task_ds_list.append(task_ds)

    few_shot_ds: Dataset = concatenate_datasets(task_ds_list)
    save_dataset(few_shot_ds, out_path)
    logger.info('Save {} examples to {}'.format(len(few_shot_ds), out_path))


if __name__ == '__main__':
    main()
