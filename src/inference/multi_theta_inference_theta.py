import os
import sys
sys.path.insert(0, 'src/')
from tqdm import tqdm
from typing import List, Dict
from datasets import Dataset, load_dataset, DownloadMode, concatenate_datasets
from typing import List, Tuple
from transformers import HfArgumentParser
import json

from config import Arguments
from logger_config import logger
from utils import save_dataset
from evaluation import BaseEval
from model_utils import build_eval_model
from inference.inference_utils import get_prompt_save_path
from llms import BaseLLM
from model_utils import build_multi_theta_llm

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
    eval_dataset: Dataset = load_dataset(
        'json', data_files='{}/{}.jsonl.gz'.format(args.data_dir, args.llm_eval_split), split='train')
    # corpus: Dataset = load_dataset(
    #     'json', data_files='{}/train.jsonl.gz'.format(args.data_dir), split='train')
    # corpus = corpus.rename_column('query_id', 'id')
    # def combine_query_answers(example):
    #     example['contents'] = example['query'] + '\n' + example['answers'][0]
    #     return example
    # corpus = corpus.map(combine_query_answers)
    
    args.llm_eval_tasks = sorted(args.llm_eval_tasks)
    # if not args.llm_eval_tasks or args.llm_eval_tasks[0] == 'all':
    #     real_llm_eval_tasks = sorted(eval_dataset.unique('task_name')) #30个任务
    #     args.llm_eval_tasks = real_llm_eval_tasks
    #     # args.llm_eval_tasks = sorted([task for task in real_llm_eval_tasks if task not in {'mnli_m', 'mnli_mm', 'wsc273'}] + ['mnli'])
    #     # args.llm_eval_tasks = sorted(eval_dataset.unique('task_name'))  #最初的任务选择
    #     logger.info('Eval all {} tasks'.format(len(real_llm_eval_tasks)))

    model: BaseEval = build_eval_model(args=args, corpus=corpus)
    llm: BaseLLM = build_multi_theta_llm(args)
    llm.cuda(args.process_index)

    task_ds_list: List[Dataset] = []
    for task_name in args.llm_eval_tasks:# args.llm_eval_tasks,,real_llm_eval_tasks
        task_ds: Dataset = eval_dataset.filter(lambda x: x['task_name'] == task_name)
        with open(args.json_path, 'r') as f:
            cluster_to_data = json.load(f)

        query_to_cluster = {}
        for cluster, data_ids in cluster_to_data.items():
            for data_id in data_ids:
                query_to_cluster[data_id] = int(cluster)

        def add_cluster(example):
            data_id = example['query_id']
            cluster = query_to_cluster.get(data_id, None)  # 如果没有找到，返回 None
            return {'cluster': cluster}
        task_ds = task_ds.map(add_cluster, desc="Adding cluster information")

        logger.info(f"Dataset with cluster column has {len(task_ds)} entries.")
        print(task_ds.column_names)  # 查看所有列名，确保 'cluster' 已添加
        print(task_ds[0])  # 查看第一个数据点，确认 'cluster' 值
        

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

        task_corpus = corpus.filter(lambda x: x['task_name'] == task_name)#同一个task内挑选2000个
        task_corpus = task_corpus.shuffle(seed=args.seed).select(range(2000)) #2000
        
        assert len(task_corpus) > 0


        all_topk_doc_ids: Dict[int, List[str]] = {}
        all_topk_scores: Dict[int, List[float]] = {}
        for k in range(args.n_cluster):
            prefix_description = ''.join([f'<{task_name}-{k}-{i}>' for i in range(args.n_prefix_tokens)])
            input_texts: List[str] = []
            output_texts: List[str] = []
            for passage in task_corpus:
                content = passage['contents']
                assert content.count('\n') == 1
                content_query, content_answer = content.split('\n', 1)
                
                if args.channel == True:
                    content = content_answer + '\n' + content_query #channel
                else:
                    content = content_query + '\n' + content_answer
                input_texts.append(content)
                output_texts.append(prefix_description)
            llm_scores: List[float] = llm.batch_score(input_texts=input_texts, output_texts=output_texts)
            assert len(llm_scores) == len(task_corpus), "Mismatch between llm_scores and task_corpus length"
            
            combined = list(zip(task_corpus['id'], llm_scores, task_corpus['contents']))
            sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)# 按得分从大到小排序
            topk_combined = sorted_combined[:args.llm_k_shot]
            single_topk_doc_ids, single_topk_scores, single_topk_contents = zip(*topk_combined)
            single_topk_doc_ids = list(single_topk_doc_ids)
            single_topk_scores = list(single_topk_scores)
            single_topk_contents = list(single_topk_contents)

            all_topk_doc_ids[k] = single_topk_doc_ids
            all_topk_scores[k] = single_topk_scores

        topk_doc_ids = []
        topk_scores = []
        dataset_clusters: List[str] = task_ds['cluster']
        for each_cluster in tqdm(dataset_clusters, desc="Processing queries"):
            topk_doc_ids.append(all_topk_doc_ids[each_cluster])
            topk_scores.append(all_topk_scores[each_cluster])

        if args.channel == True:
            input_prompts: List[str] = [model.channel_get_prompt_by_doc_ids(doc_ids) for doc_ids in topk_doc_ids]
        else:
            input_prompts: List[str] = [model.get_prompt_by_doc_ids(doc_ids) for doc_ids in topk_doc_ids]

        # input_prompts: List[List[str]] = [input_prompts_single for _ in range(len(task_ds))]
        # topk_doc_ids: List[List[str]] = [single_topk_doc_ids for _ in range(len(task_ds))]
        # topk_scores: List[List[float]] = [single_topk_scores for _ in range(len(task_ds))]

        task_ds = task_ds.add_column('input_prompt', input_prompts)#前面已经确保了task_ds中有query信息
        task_ds = task_ds.add_column('topk_doc_ids', topk_doc_ids)
        task_ds = task_ds.add_column('topk_scores', topk_scores)
        #删除doc_ids，和doc_scores列
        # task_ds = task_ds.remove_columns(['doc_ids', 'doc_scores'])
        task_ds_list.append(task_ds)

    few_shot_ds: Dataset = concatenate_datasets(task_ds_list)
    save_dataset(few_shot_ds, out_path)
    logger.info('Save {} examples to {}'.format(len(few_shot_ds), out_path))


if __name__ == '__main__':
    main()
