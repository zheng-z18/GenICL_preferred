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

import debugpy
try:
    debugpy.listen(("localhost", 9609))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass


parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]

def main():
    out_path: str = get_prompt_save_path(args=args)
    if os.path.exists(out_path):
        logger.info('Prompt file {} exists. Skip.'.format(out_path))
        return

    # corpus: Dataset = load_dataset(
    #     'json', data_files='{}/passages.jsonl.gz'.format(args.data_dir), split='train')
    eval_dataset: Dataset = load_dataset( #data/tasks/e5-base_test.jsonl.gz
        'json', data_files='{}/{}.jsonl.gz'.format(args.data_dir, args.llm_eval_split), split='train', download_mode='force_redownload')
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

    # model: BaseEval = build_eval_model(args=args, corpus=corpus)
    llm: BaseLLM = build_extend_llm(args)
    llm.cuda(args.process_index)

    task_ds_list: List[Dataset] = []
    for task_name in args.llm_eval_tasks:# args.llm_eval_tasks,,real_llm_eval_tasks
        task_ds: Dataset = eval_dataset.filter(lambda x: x['task_name'] == task_name)
        
        # if task_name in ['mnli_m', 'mnli_mm']:
        #     task_name = 'mnli'
        # if task_name == 'wsc273':
        #     task_name = 'wsc'
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

        # task_corpus = corpus.filter(lambda x: x['task_name'] == task_name)#同一个task内挑选
        # task_corpus = task_corpus.shuffle(seed=42).select(range(100)) #从中随机挑选100个
        # assert len(task_corpus) > 0

        # input_texts: List[str] = []
        # output_texts: List[str] = []
        # for passage in task_corpus:
        #     content = passage['contents']
        #     assert content.count('\n') == 1
        #     content_query, content_answer = content.split('\n', 1) #channel
        #     content = content_answer + '\n' + content_query #channel
        #     # content_query = f'{prefix_description}\n\n{content_query}'######使用prefix_description
        #     input_texts.append(content)
        #     output_texts.append(prefix_description)
        
        # llm_scores: List[float] = llm.batch_score(input_texts=input_texts, output_texts=output_texts)
        # assert len(llm_scores) == len(task_corpus), "Mismatch between llm_scores and task_corpus length"
        
        # combined = list(zip(task_corpus['id'], llm_scores, task_corpus['contents']))
        # sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)# 按得分从大到小排序
        # topk_combined = sorted_combined[:args.llm_k_shot]
        # single_topk_doc_ids, single_topk_scores, single_topk_contents = zip(*topk_combined)
        # single_topk_doc_ids = list(single_topk_doc_ids)
        # single_topk_scores = list(single_topk_scores)
        # single_topk_contents = list(single_topk_contents)

        # input_prompts_single: List[str] = model.channel_get_prompt_by_doc_ids(single_topk_doc_ids)
        

        # input_prompts_single: List[str] = model.channel_get_prompt_by_train(single_topk_doc_ids, task_corpus)
        # input_prompts_single = "sustain the buoyant energy level of the film 's city beginnings into its country conclusion '\nPositive\n\ndoes n't trust laughs -- and does n't conjure proper respect for followers of the whole dead-undead genre , who deserve more from a vampire pic than a few shrieky special effects .\nNegative\n\nremarkable procession of sweeping pictures that have reinvigorated the romance genre\nPositive\n\neven on its own ludicrous terms , the sum of all fears generates little narrative momentum , and invites unflattering comparisons to other installments in the ryan series .\nNegative"
        # input_prompts_single = "\n\n\nnegative\nReview: ( jackson and bledel ) seem to have been picked not for their acting chops , but for their looks and appeal to the pre-teen crowd .\n\n\npositive\nReview: all in all , brown sugar is a satisfying well-made romantic comedy that 's both charming and well acted .\n\n\nnegative\nReview: seemingly a vehicle to showcase the canadian 's inane ramblings , stealing harvard is a smorgasbord of soliloquies about nothing delivered by the former mr. drew barrymore .\n\n\npositive\nReview: superb production values & christian bale 's charisma make up for a derivative plot ."
        input_prompts_single = "\"Bombs Hit Three Argentine Bank Branches, One Dead &lt;p&gt;&lt;/p&gt;&lt;p&gt; By Mary Milliken&lt;/p&gt;&lt;p&gt; BUENOS AIRES, Argentina (Reuters) - Bombs exploded at threebanks in the Argentine capital on Wednesday, including twobranches of U.S. giant Citibank, killing one security guard andinjuring a bomb squad officer, officials said.&lt;/p&gt;\" What is this text about? World, Sports, Business, or Technology?\nWorld\n\n\n\"Boeing Probe Could Spur Rival Lawsuits  WASHINGTON (Reuters) - A confession by the former No. 2 Air  Force arms buyer that she steered contracts to her future  employer, Boeing Co. &lt;A HREF=\"http://www.investor.reuters.com/FullQuote.aspx?ticker=BA.N target=/stocks/quickinfo/fullquote\"&gt;BA.N&lt;/A&gt;, could spur legal challenges to old  contract awards, legal experts said on Monday.\" What is this text about? World, Sports, Business, or Technology?\nBusiness\n\n\n\"Amgen Quarterly Profit Falls  NEW YORK (Reuters) - Amgen Inc. &lt;A HREF=\"http://www.investor.reuters.com/FullQuote.aspx?ticker=AMGN.O target=/stocks/quickinfo/fullquote\"&gt;AMGN.O&lt;/A&gt;, the world's  biggest biotechnology company, on Wednesday reported a lower  third-quarter profit related to the acquisition of another  biotech company.\" What is this text about? World, Sports, Business, or Technology?\nBusiness\n\n\n\"Where the Game Is the Medium The co-founder of game producer WildTangent Inc. is a booming bear of a man who clobbers you with one grand pronouncement after another. &lt;FONT face=\"verdana,MS Sans Serif,arial,helvetica\" size=\"-2\" color=\"#666666\"&gt;&lt;B&gt;-Leslie Walker&lt;/B&gt;&lt;/FONT&gt;\" What is this text about? World, Sports, Business, or Technology?\nTechnology\n\n\n\"Tyco Raises Dividend  BOSTON (Reuters) - Tyco International Inc. &lt;A HREF=\"http://www.investor.reuters.com/FullQuote.aspx?ticker=TYC.N target=/stocks/quickinfo/fullquote\"&gt;TYC.N&lt;/A&gt;        on  Thursday raised its dividend for the first time in five years,  the latest indication of change at the once scandal-tarnished  diversified manufacturer.\" What is this text about? World, Sports, Business, or Technology?\nBusiness\n\n\n\"United asks court to void contracts Says it needs to save an additional \\$725M and also asks for pension relief; strikes loom. NEW YORK (Reuters) - The parent of United Airlines Inc.\" What is this text about? World, Sports, Business, or Technology?\nBusiness\n\n\n\"Stocks Seen Higher Before Jobs Data  NEW YORK (Reuters) - U.S. stock futures surged Friday  morning on the back of Intel Corp.'s &lt;A HREF=\"http://www.investor.reuters.com/FullQuote.aspx?ticker=INTC.O target=/stocks/quickinfo/fullquote\"&gt;INTC.O&lt;/A&gt; raised quarterly  sales forecast and lower oil prices, but strategists said the  market's early direction would be dictated by the monthly  employment data due out before the start of trading.\" What is this text about? World, Sports, Business, or Technology?\nBusiness\n\n\n\"Another citizen of US abducted BAGHDAD, Iraq - Gunmen kidnapped a Lebanese-American businessman and militants released a tape yesterday showing the beheading of an Iraqi officer as a warning to those who deal with  quot;the infidel quot; Americans.\" What is this text about? World, Sports, Business, or Technology?\nWorld"
        input_prompts: List[List[str]] = [input_prompts_single for _ in range(len(task_ds))]
        topk_doc_ids: List[List[str]] = [0 for _ in range(len(task_ds))]
        topk_scores: List[List[float]] = [0 for _ in range(len(task_ds))]

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
