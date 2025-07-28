import os
import sys
import numpy as np
import bm25s
from tqdm import tqdm
from datasets import Dataset, load_dataset
from typing import Dict, List, Tuple
from transformers import HfArgumentParser
import multiprocessing
import json
import pickle
# from Stemmer import Stemmer  
from config import Arguments
from logger_config import logger
from utils import save_dataset
from data_utils import save_to_readable_format

import debugpy
try:
    debugpy.listen(("localhost", 9609))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

# 初始化多进程
# multiprocessing.set_start_method('spawn', force=True)  # JAX 与多进程 fork 不兼容问题

parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]

def main():
    out_path = f"{args.output_dir}/bm25_{args.search_split}.jsonl.gz"
    if os.path.exists(out_path):
        logger.info(f"输出文件 {out_path} 已存在，跳过。")
        return

    data_path = f"{args.data_dir}/{args.search_split}.jsonl.gz"
    assert os.path.exists(data_path), f"数据文件 {data_path} 不存在。"
    dataset = load_dataset('json', data_files=data_path, split='train')
    if args.dry_run:
        dataset = dataset.shuffle(seed=args.seed).select(range(100))
    logger.info(f"从 {data_path} 加载了 {len(dataset)} 条数据")

    corpus_path = f"{args.data_dir}/passages.jsonl.gz"
    corpus = load_dataset('json', data_files=corpus_path, split='train')
    logger.info(f"从 {corpus_path} 加载了 {len(corpus)} 条候选文档")

    dataset_query = dataset['query']
    corpus_contents = corpus['contents']
    retriever_path = "data/bm25/bm25_test/retriever_bm25"

    # tokenized_corpus = bm25s.tokenize(corpus['contents'])
    # retriever = bm25s.BM25()
    # retriever.index(tokenized_corpus)
    # retriever.save(retriever_path)

    retriever = bm25s.BM25.load(retriever_path)

    for each_query in tqdm(dataset['query'], desc="Processing queries"):
        query_tokens = bm25s.tokenize(each_query)
        results, scores = retriever.retrieve(query_tokens, k=args.search_topk)

        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]
            print(f"Rank {i+1} (score: {score:.2f}): {doc}", corpus_contents[doc])












    results.sort(key=lambda x: x[0])
    topk_score_doc_ids = [r[1] for r in results]
    all_contents = corpus['contents']

    def _map_func(example: Dict, idx: int) -> Dict:
        score_doc_ids = topk_score_doc_ids[idx]
        score_doc_ids = [
            t for t in score_doc_ids
            if not all_contents[int(t[1])].startswith(example['query'])
        ]
        return {
            'doc_ids': [doc_id for _, doc_id in score_doc_ids],
            'doc_scores': [round(doc_score, 4) for doc_score, _ in score_doc_ids],
        }

    logger.info("开始为数据集添加BM25检索结果...")
    dataset = dataset.map(
        _map_func,
        with_indices=True,
        num_proc=1,
        desc='Add top-k BM25 candidates'
    )

    # 保留有至少2个候选文档的样本
    dataset = dataset.filter(lambda example: len(example['doc_ids']) > 1)
    logger.info(f"过滤后数据集大小为：{len(dataset)}")

    # 保存数据集
    save_dataset(dataset, out_path, shuffle='train' in args.search_split)
    logger.info(f"数据集已保存至 {out_path}")

    # 保存可读格式文件
    save_to_readable_format(in_path=out_path, corpus=corpus, shuffle=True)
    logger.info("可读格式文件已保存完成。")

if __name__ == '__main__':
    main()
