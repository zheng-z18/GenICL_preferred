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
from model_utils import build_MT_llm
from accelerate import Accelerator

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
    accelerator = Accelerator()

    out_path: str = get_prompt_save_path(args=args)
    if os.path.exists(out_path):
        logger.info('Prompt file {} exists. Skip.'.format(out_path))
        return

    corpus: Dataset = load_dataset(
        'json', data_files='{}/passages.jsonl.gz'.format(args.data_dir), split='train')
    # eval_dataset: Dataset = load_dataset(
    #     'json', data_files='{}/{}.jsonl.gz'.format(args.data_dir, args.llm_eval_split), split='train')
    retriever_name = os.path.basename(args.model_name_or_path)
    eval_dataset: Dataset = load_dataset(
        'json', data_files='{}/{}_{}.jsonl.gz'.format(args.data_dir, retriever_name, args.llm_eval_split), split='train')
    
    args.llm_eval_tasks = sorted(args.llm_eval_tasks)

    # model: BaseEval = build_eval_model(args=args, corpus=corpus)
    # llm: BaseLLM = build_extend_llm(args)
    llm: BaseLLM = build_MT_llm(args, accelerator=accelerator)


    task_ds_list: List[Dataset] = []
    for task_name in args.llm_eval_tasks:# args.llm_eval_tasks,,real_llm_eval_tasks
        task_ds: Dataset = eval_dataset.filter(lambda x: x['task_name'] == task_name)

        if len(task_ds) > 1000:
            task_ds = task_ds.shuffle(seed=args.seed).select(range(1000))

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

        topk_doc_ids = []
        topk_scores = []

        for each_data in tqdm(task_ds, desc="Processing data", unit="item"):
            input_texts: List[str] = []
            output_texts: List[str] = []
            query = each_data['query']
            doc_ids = each_data['doc_ids'][:100] ###########每个query跑多少个doc
            for doc_id in doc_ids:
                content = corpus[int(doc_id)]['contents']
                input_texts.append(query)
                output_texts.append(content)
            
            with accelerator.autocast():
                llm_scores: List[float] = llm.batch_score(input_texts=input_texts, output_texts=output_texts)
            # assert len(llm_scores) == len(task_corpus), "Mismatch between llm_scores and task_corpus length"
            
            combined = list(zip(doc_ids, llm_scores)) #task_corpus['contents']
            sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)# 按得分从大到小排序
            topk_combined = sorted_combined[:args.llm_k_shot]

            # topk_combined = topk_combined[::-1]  # 反转列表顺序,将得分最大的放在最后

            single_topk_doc_ids, single_topk_scores = zip(*topk_combined)
            single_topk_doc_ids = list(single_topk_doc_ids)
            single_topk_scores = list(single_topk_scores)
            topk_doc_ids.append(single_topk_doc_ids)
            topk_scores.append(single_topk_scores)

        def get_prompt_by_doc_ids(doc_ids: List[str]) -> str:
            prompts = []
            for doc_id in doc_ids:
                corpus_entry = corpus[int(doc_id)]
                if int(corpus_entry['id']) != int(doc_id):# 验证 doc_id 是否与 corpus 中的 id 对应
                    raise ValueError(f"doc_id {doc_id} does not match corpus entry id {corpus_entry['id']}")
                prompts.append(corpus_entry['contents'])
            return '\n\n'.join(prompts)
        input_prompts: List[str] = [get_prompt_by_doc_ids(doc_ids) for doc_ids in topk_doc_ids]
        # input_prompts: List[str] = [model.get_prompt_by_doc_ids(doc_ids) for doc_ids in topk_doc_ids]
        task_ds = task_ds.add_column('input_prompt', input_prompts)
        task_ds = task_ds.add_column('topk_doc_ids', topk_doc_ids)
        task_ds = task_ds.add_column('topk_scores', topk_scores)
        task_ds_list.append(task_ds)

    few_shot_ds: Dataset = concatenate_datasets(task_ds_list)


    if accelerator.is_local_main_process:
        save_dataset(few_shot_ds, out_path)
        logger.info('Save {} examples to {}'.format(len(few_shot_ds), out_path))
    

if __name__ == '__main__':
    main()
