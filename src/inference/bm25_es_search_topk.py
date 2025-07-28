import os
import sys
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
sys.path.insert(0, 'src/')
from tqdm import tqdm
from datasets import Dataset, load_dataset
from typing import Dict, List, Tuple
from transformers import HfArgumentParser
import multiprocessing
import json
from config import Arguments
from logger_config import logger
from utils import save_dataset
from data_utils import save_to_readable_format

# 引入您提供的ElasticSearchBM25类
from ElasticSearchBM25 import ElasticSearchBM25

parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]

# 全局变量，用来存储 Elasticsearch 服务的基本信息
ES_HOST = "http://localhost"
ES_PORT_HTTP = "9200"
# 您可以根据需要调整索引名和是否需要reindexing等参数
ES_INDEX_NAME = "my_corpus_index"

def init_global_es_bm25(corpus_map):
    """
    初始化ElasticSearchBM25对象。
    corpus_map是一个dict: {doc_id: doc_content}的映射，用来创建索引。
    """
    global es_bm25
    es_bm25 = ElasticSearchBM25(
        corpus_map,
        index_name=ES_INDEX_NAME,
        reindexing=True,  # 如果希望每次运行都重新建索引，可以设为True
        host="localhost",
        port_http=ES_PORT_HTTP
    )

def process_query(args_tuple):
    idx, query_tokens, search_topk = args_tuple
    # 在子进程中也创建ElasticSearchBM25实例（如果数据量和性能允许）
    # 或者您可以选择在主进程中完成查询，再考虑使用其他并行方式。
    # 这里为了展示简化逻辑，假设corpus_map在全局中可访问，否则您需要传入。
    global corpus_map
    es_bm25_local = ElasticSearchBM25(
        corpus_map,
        # index_name=ES_INDEX_NAME,
        reindexing=False,  # 索引已经创建，无需重复
        # host="localhost",
        # port_http=ES_PORT_HTTP
    )
    query = " ".join(query_tokens)
    # 查询
    rank, scores = es_bm25_local.query(query, topk=search_topk, return_scores=True)
    # rank是 {doc_id: doc_content}，scores是 {doc_id: score}
    # 排序结果已经由ES给出，可以直接使用
    # 返回形式与之前BM25Okapi一致，即 [(score, doc_id), ...]
    result_list = []
    for doc_id in rank.keys():
        result_list.append((scores[doc_id], doc_id))
    return idx, result_list

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

    # 对corpus进行分词或直接使用
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

    # corpus_map: {doc_id: doc_content}
    # 因为ElasticSearchBM25需要doc_id->doc_content映射，这里以索引作为id
    global corpus_map
    corpus_map = {str(i): c for i, c in enumerate(corpus['contents'])}
    
    # 初始化ElasticSearchBM25，建立索引
    init_global_es_bm25(corpus_map)

    logger.info(f"使用ElasticSearchBM25为每个查询并行检索前 {args.search_topk} 个候选文档...")
    work_args = [(idx, query_tokens, args.search_topk) for idx, query_tokens in enumerate(tokenized_queries)]

    # 根据需要设置进程数
    with multiprocessing.Pool(processes=None) as pool:
        results = list(tqdm(pool.imap_unordered(process_query, work_args),
                            total=len(work_args),
                            desc="Processing queries in parallel"))

    # 将并行处理的结果根据原索引排序
    results.sort(key=lambda x: x[0]) 
    topk_score_doc_ids: List[List[Tuple[float, str]]] = [r[1] for r in results]

    all_contents: List[str] = corpus['contents']

    def _map_func(example: Dict, idx: int) -> Dict:
        score_doc_ids: List[Tuple[float, str]] = topk_score_doc_ids[idx]
        # 根据需要过滤
        score_doc_ids = [
            t for t in score_doc_ids
            if not all_contents[int(t[1])].startswith(example['query'])
        ]
        return {
            'doc_ids': [doc_id for _, doc_id in score_doc_ids],
            'doc_scores': [round(doc_score, 4) for doc_score, _ in score_doc_ids],
        }

    logger.info("开始为数据集添加检索结果...")
    dataset = dataset.map(
        _map_func,
        with_indices=True,
        num_proc=1,
        desc='Add top-k BM25 candidates'
    )

    # 过滤掉没有足够候选文档的样本
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
