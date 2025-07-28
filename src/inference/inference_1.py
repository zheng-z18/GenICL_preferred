import os
import sys
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
        'json', data_files='{}/{}_{}.jsonl.gz'.format(args.data_dir,os.path.basename(args.model_name_or_path), args.llm_eval_split), split='train')
    args.llm_eval_tasks = sorted(args.llm_eval_tasks)
    if not args.llm_eval_tasks or args.llm_eval_tasks[0] == 'all':
        real_llm_eval_tasks = sorted(eval_dataset.unique('task_name')) #30个任务
        args.llm_eval_tasks = real_llm_eval_tasks
        # args.llm_eval_tasks = sorted([task for task in real_llm_eval_tasks if task not in {'mnli_m', 'mnli_mm', 'wsc273'}] + ['mnli'])
        # args.llm_eval_tasks = sorted(eval_dataset.unique('task_name'))  #最初的任务选择
        logger.info('Eval all {} tasks'.format(len(real_llm_eval_tasks)))

    model: BaseEval = build_eval_model(args=args, corpus=corpus)
    llm: BaseLLM = build_extend_llm(args)
    llm.cuda(args.process_index)

    task_ds_list: List[Dataset] = []
    for task_name in args.llm_eval_tasks:# args.llm_eval_tasks,,real_llm_eval_tasks
        task_ds: Dataset = eval_dataset.filter(lambda x: x['task_name'] == task_name)
        # task_ds = task_ds.select(range(len(task_ds)//20))  #for degug
        
        if task_name in ['mnli_m', 'mnli_mm']:
            task_name = 'mnli'
        if task_name == 'wsc273':
            task_name = 'wsc'
        prefix_description = ''.join([f'<{task_name}-{i}>' for i in range(args.n_prefix_tokens)])

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
        
        dataset_queries: List[str] = task_ds['query']
        dataset_doc_ids: List[List[str]] = task_ds['doc_ids']

        input_texts: List[str] = []
        output_texts: List[str] = []
        for query, doc_ids in zip(dataset_queries, dataset_doc_ids):
        # for query, doc_ids in zip(task_ds['query'], task_ds['doc_ids']):
            # if len(demo_query_description_sentences) > 300:##for debug
            #     break
            for doc_id in doc_ids:
                content = corpus[int(doc_id)]['contents'] 
                content_query, content_answer = content.split('\n', 1)
                content_query = '{}\n\n{}'.format(prefix_description, content_query)######使用prefix
                input_texts.append(content_query)
                output_texts.append(content_answer)
        llm_scores: List[float] = llm.batch_score(input_texts=input_texts, output_texts=output_texts)

        # import random
        # score_perplexity = [random.randint(0, 1000) for _ in range(len(task_ds['doc_ids']) * 100)]
        logger.info('doc_ids length: {}'.format(len(task_ds['doc_ids'])))
        logger.info('score_perplexity length: {}'.format(len(llm_scores)))
        
        start_idx = 0
        dataset_doc_scores: List[List[int]] = []
        for i, doc_ids in enumerate(tqdm(dataset_doc_ids, desc='reconstruct doc_scores')):
            end_idx = start_idx + len(doc_ids)
            # if start_idx >= len(score_perplexity):##for debug
            #     break
            assert start_idx < len(llm_scores)
            dataset_doc_scores.append(llm_scores[start_idx:end_idx]) #获得检索到的各个样本的得分，
            start_idx = end_idx

        topk_doc_ids = []
        topk_scores = []
        for ids, scores in zip(dataset_doc_ids, dataset_doc_scores):
            combined = list(zip(ids, scores))
            sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)#从大到小
            top_k = sorted_combined[:args.llm_k_shot] #取最大值
            single_topk_doc_ids, single_topk_scores = zip(*top_k) if top_k else ([], [])
            topk_doc_ids.append(list(single_topk_doc_ids))  # 转换为列表并添加到 all_topk_doc_ids
            topk_scores.append(list(single_topk_scores))  # 转换为列表并添加到 all_topk_scores
        
        input_prompts: List[str] = [model.get_prompt_by_doc_ids(doc_ids) for doc_ids in topk_doc_ids]#根据doc_id获取corpus的content
        task_ds = task_ds.add_column('input_prompt', input_prompts)#前面已经确保了task_ds中有query信息
        task_ds = task_ds.add_column('topk_doc_ids', topk_doc_ids)
        task_ds = task_ds.add_column('topk_scores', topk_scores)
        #删除doc_ids，和doc_scores列
        task_ds = task_ds.remove_columns(['doc_ids', 'doc_scores'])
        task_ds_list.append(task_ds)

    few_shot_ds: Dataset = concatenate_datasets(task_ds_list)
    save_dataset(few_shot_ds, out_path)
    logger.info('Save {} examples to {}'.format(len(few_shot_ds), out_path))


if __name__ == '__main__':
    main()
