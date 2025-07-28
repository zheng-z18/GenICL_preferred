import os
import sys
sys.path.insert(0, 'src/')

from datasets import Dataset, load_dataset, DownloadMode, concatenate_datasets
from typing import List, Tuple
from transformers import HfArgumentParser

from config import Arguments
from logger_config import logger
from utils import save_dataset
from evaluation import BaseEval
from model_utils import build_eval_model
from inference.inference_utils import get_prompt_save_path

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
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
        'json', data_files='{}/{}.jsonl.gz'.format(args.data_dir, args.llm_eval_split), split='train')

    if not args.llm_eval_tasks or args.llm_eval_tasks[0] == 'all':
        args.llm_eval_tasks = sorted(eval_dataset.unique('task_name')) #30个任务
        logger.info('Eval all {} tasks'.format(len(args.llm_eval_tasks)))

    model: BaseEval = build_eval_model(args=args, corpus=corpus)

    task_ds_list: List[Dataset] = []
    for task_name in args.llm_eval_tasks:
        task_ds: Dataset = eval_dataset.filter(lambda x: x['task_name'] == task_name)

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

        queries: List[str] = task_ds['query'] #获取该任务下所有样本的query
        # Use a larger k in case we retrieve the docs from other tasks
        topk_score_doc_ids: List[List[Tuple[float, str]]] = model.get_topk_score_doc_ids(
            queries, k=5 * args.llm_k_shot, task_names=task_ds['task_name']
        )
        # The most relevant doc should be close to the test example #获取前args.llm_k_shot个
        topk_score_doc_ids = [score_doc_ids[:args.llm_k_shot][::-1] for score_doc_ids in topk_score_doc_ids] #[::-1]反转
        topk_doc_ids: List[List[str]] = [  #获得doc_id，忽略score
            [doc_id for _, doc_id in score_doc_ids] for score_doc_ids in topk_score_doc_ids
        ]
        topk_scores: List[List[float]] = [ #获得score，忽略doc_id
            [round(score, 4) for score, _ in score_doc_ids] for score_doc_ids in topk_score_doc_ids
        ]
        input_prompts: List[str] = [model.get_prompt_by_doc_ids(doc_ids) for doc_ids in topk_doc_ids]#根据doc_id获取corpus的content
        task_ds = task_ds.add_column('input_prompt', input_prompts)
        task_ds = task_ds.add_column('topk_doc_ids', topk_doc_ids)
        task_ds = task_ds.add_column('topk_scores', topk_scores)
        task_ds_list.append(task_ds)

    few_shot_ds: Dataset = concatenate_datasets(task_ds_list)
    save_dataset(few_shot_ds, out_path)
    logger.info('Save {} examples to {}'.format(len(few_shot_ds), out_path))


if __name__ == '__main__':
    main()
