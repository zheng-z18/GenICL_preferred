import os
import sys
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
sys.path.insert(0, 'src/')
from tqdm import tqdm
from datasets import Dataset, load_dataset
from typing import Dict, List, Tuple
from transformers import HfArgumentParser
import multiprocessing  # 新增的导入
import json
from config import Arguments
from logger_config import logger
from utils import save_dataset
from data_utils import save_to_readable_format

parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]

def init_global_bm25(bm25_obj):
    # 在子进程中初始化全局bm25变量
    global bm25_global
    bm25_global = bm25_obj

def process_query(args_tuple):
    # 子进程中处理单个查询的函数
    idx, query_tokens, search_topk = args_tuple
    # 使用全局的bm25对象进行检索
    scores = bm25_global.get_scores(query_tokens)
    topk_indices = np.argsort(scores)[::-1][:search_topk]
    topk_scores = scores[topk_indices]
    topk_ids = [str(i) for i in topk_indices]
    return idx, list(zip(topk_scores, topk_ids))

def main():
    out_path: str = f"{args.output_dir}/bm25_{args.search_split}.jsonl.gz"
    if os.path.exists(out_path):
        logger.info(f"输出文件 {out_path} 已存在，跳过。")
        return

    data_path: str = f"{args.data_dir}/{args.search_split}.jsonl.gz"
    assert os.path.exists(data_path), f"数据文件 {data_path} 不存在。"
    dataset: Dataset = load_dataset('json', data_files=data_path, split='train')
    if args.dry_run:
        dataset = dataset.shuffle(seed=args.seed).select(range(100))
    logger.info(f"从 {data_path} 加载了 {len(dataset)} 条数据")

    corpus_path: str = f"{args.data_dir}/passages.jsonl.gz"
    corpus: Dataset = load_dataset('json', data_files=corpus_path, split='train')
    logger.info(f"从 {corpus_path} 加载了 {len(corpus)} 条候选文档")


    corpus_token_path = f"{args.output_dir}/bm25_{args.search_split}/tokenized_corpus.json"
    logger.info(f"corpus_token_path: {corpus_token_path}")
    if os.path.exists(corpus_token_path):
        with open(corpus_token_path, 'r', encoding='utf-8') as f:
            tokenized_corpus = json.load(f)
        logger.info("已从缓存文件加载分词后的语料。")
    else:
        tokenized_corpus = [word_tokenize(doc.lower()) for doc in tqdm(corpus['contents'], desc="Tokenizing corpus")]
        os.makedirs(os.path.dirname(corpus_token_path), exist_ok=True)
        with open(corpus_token_path, 'w', encoding='utf-8') as f:
            json.dump(tokenized_corpus, f, ensure_ascii=False)
        logger.info("语料分词完成并已保存缓存文件。")

    queries_token_path = f"{args.output_dir}/bm25_{args.search_split}/tokenized_queries.json"
    logger.info(f"queries_token_path: {queries_token_path}")
    if os.path.exists(queries_token_path):
        with open(queries_token_path, 'r', encoding='utf-8') as f:
            tokenized_queries = json.load(f)
        logger.info("已从缓存文件加载分词后的查询。")
    else:
        tokenized_queries = [word_tokenize(q.lower()) for q in tqdm(dataset['query'], desc="Tokenizing queries")]
        os.makedirs(os.path.dirname(queries_token_path), exist_ok=True)
        with open(queries_token_path, 'w', encoding='utf-8') as f:
            json.dump(tokenized_queries, f, ensure_ascii=False)
        logger.info("查询分词完成并已保存缓存文件。")

    bm25 = BM25Okapi(tokenized_corpus)
    # 使用多进程并行检索Top-K文档
    logger.info(f"使用BM25为每个查询并行检索前 {args.search_topk} 个候选文档...")
    work_args = [(idx, query_tokens, args.search_topk) for idx, query_tokens in enumerate(tokenized_queries)]

    # 根据机器核数可适当调整进程数，processes=None默认使用所有CPU核
    with multiprocessing.Pool(processes=None, initializer=init_global_bm25, initargs=(bm25,)) as pool:
        results = list(tqdm(pool.imap_unordered(process_query, work_args),
                            total=len(work_args),
                            desc="Processing queries in parallel"))

    # 将并行处理的结果按照原来顺序排序
    results.sort(key=lambda x: x[0]) 
    topk_score_doc_ids: List[List[Tuple[float, str]]] = [r[1] for r in results]

    all_contents: List[str] = corpus['contents']

    def _map_func(example: Dict, idx: int) -> Dict:
        score_doc_ids: List[Tuple[float, str]] = topk_score_doc_ids[idx]
        # 排除掉与查询本身内容重复的文档（防止检索出的是query本身）
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
