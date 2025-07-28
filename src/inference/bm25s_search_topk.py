import os
import sys
import numpy as np
from typing import Dict, List, Tuple
from datasets import Dataset, load_dataset
from transformers import HfArgumentParser
from tqdm import tqdm

from config import Arguments
from logger_config import logger
from utils import save_dataset
from data_utils import save_to_readable_format
from evaluation import BaseEval
from model_utils import build_eval_model, parse_model_id
import bm25s

import debugpy
try:
    debugpy.listen(("localhost", 9609))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    assert args.do_search, 'This script is only for search mode.'

    # Parse the model ID (if needed for BM25, otherwise this can be skipped)
    out_path: str = f"{args.output_dir}/bm25_{args.search_split}.jsonl.gz"
    if os.path.exists(out_path):
        logger.info(f"Output file {out_path} already exists. Skipping.")
        return

    # Load the main dataset
    data_path: str = f"{args.data_dir}/{args.search_split}.jsonl.gz"
    assert os.path.exists(data_path), f"Data file {data_path} does not exist."
    dataset: Dataset = load_dataset('json', data_files=data_path, split='train')
    if args.dry_run:
        dataset = dataset.shuffle(seed=args.seed).select(range(100))
    # dataset: Dataset = dataset.filter(lambda x: x['task_name'] == "rte") #######去除gigaword task
    logger.info(f"Loaded {len(dataset)} examples from {data_path}")

    # Load the corpus dataset
    corpus_path: str = f"{args.data_dir}/passages.jsonl.gz"
    corpus: Dataset = load_dataset('json', data_files=corpus_path, split='train')
    # corpus: Dataset = corpus.filter(lambda x: x['task_name'] != "gigaword") #######去除gigaword task
    logger.info(f"Loaded {len(corpus)} candidate documents from {corpus_path}")
    corpus_contents: List[str] = corpus['contents']

    # Initialize BM25 retriever
    retriever_path = "data/bm25/bm25_test/retriever_bm25"
    if not os.path.exists(retriever_path):
        # If retriever is not saved, initialize and save it
        tokenized_corpus = bm25s.tokenize(corpus['contents'])
        retriever = bm25s.BM25()
        retriever.index(tokenized_corpus)
        retriever.save(retriever_path)
        logger.info(f"BM25 retriever indexed and saved to {retriever_path}")
    else:
        retriever = bm25s.BM25.load(retriever_path)
        logger.info(f"BM25 retriever loaded from {retriever_path}")

    # Retrieve top-k documents for all queries
    logger.info(f"Retrieving top {args.search_topk} documents for each query using BM25.")
    topk_score_doc_ids: List[List[Tuple[float, int]]] = []
    for query in tqdm(dataset['query'], desc="Retrieving top-k documents for queries"):
        query_tokens = bm25s.tokenize(query)
        results, scores = retriever.retrieve(query_tokens, k=args.search_topk)  #doc, score = results[0, i], scores[0, i]
        # Assuming results and scores are numpy arrays with shape (1, k)
        # Convert doc indices to integers and pair with scores
        topk = [(float(scores[0, i]), results[0, i]) for i in range(len(results[0]))]
        topk_score_doc_ids.append(topk)

        # for i in range(results.shape[1]):
        #     doc, score = results[0, i], scores[0, i]
        #     print(f"Rank {i+1} (score: {score:.2f}): {doc}", corpus_contents[doc])    

    def _map_func(example: Dict, idx: int) -> Dict:
        """
        Adds 'doc_ids' and 'doc_scores' to each example in the dataset.

        Args:
            example (Dict): A single example from the dataset.
            idx (int): The index of the example.

        Returns:
            Dict: A dictionary with 'doc_ids' and 'doc_scores' added.
        """
        score_doc_ids: List[Tuple[float, int]] = topk_score_doc_ids[idx]

        # Optionally exclude the example itself if the corpus contains the query
        # Uncomment the following lines if you need to exclude the query from the results
        query_text = example['query']
        score_doc_ids = [
            (score, doc_id) for score, doc_id in score_doc_ids
            if not corpus_contents[doc_id].startswith(query_text)
        ]

        # Convert doc indices to string IDs if necessary
        # Assuming doc IDs are strings; adjust if they are integers or another format
        return {
            'doc_ids': [str(doc_id) for _, doc_id in score_doc_ids],
            'doc_scores': [round(score, 4) for score, _ in score_doc_ids],
        }

    # Apply the map function to add 'doc_ids' and 'doc_scores'
    logger.info("Adding 'doc_ids' and 'doc_scores' to the dataset.")
    dataset = dataset.map(
        _map_func,
        with_indices=True,
        num_proc=1,
        desc="Adding top-k candidates"
    )

    # Optionally filter out examples that do not have enough retrieved documents
    dataset = dataset.filter(
        lambda example: len(example['doc_ids']) >= 1,
        desc="Filtering examples with insufficient documents"
    )
    logger.info(f"Dataset after adding candidates has {len(dataset)} examples.")

    # Save the updated dataset
    save_dataset(dataset, out_path, shuffle='train' in args.search_split)
    logger.info(f"Saved the updated dataset to {out_path}")

    # Convert to a readable format if needed
    save_to_readable_format(in_path=out_path, corpus=corpus, shuffle=True)
    logger.info("Converted the dataset to a readable format.")

if __name__ == '__main__':
    main()
