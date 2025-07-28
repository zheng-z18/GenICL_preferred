import os
import tqdm
import torch
import json
import numpy as np

from typing import List, Dict, Union
from datasets import load_dataset, concatenate_datasets
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans

from models.simple_encoder import SimpleEncoder
from logger_config import logger
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.utils.logging import enable_explicit_format, set_verbosity_info, set_verbosity_warning
from logger_config import logger
from config import Arguments
import logging

def _common_setup(args: Arguments):
    set_verbosity_info()
    if args.process_index > 0:
        logger.setLevel(logging.WARNING)
        set_verbosity_warning()
    enable_explicit_format()
    set_seed(args.seed)

def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    _common_setup(args)
    logger.info('Args={}'.format(str(args)))
    args.llm_eval_tasks = sorted(args.llm_eval_tasks)
    
    # Load datasets
    train_dataset = load_dataset('json', data_files='{}/train.jsonl.gz'.format(args.data_dir), split='train')
    # Apply any necessary filtering to each dataset
    train_dataset = train_dataset.filter(lambda x: x['task_name'] in args.llm_eval_tasks[0].split())
    logger.info(f"Filtered Train Dataset length: {len(train_dataset)}")

    # Add a column to each dataset to identify which dataset it is
    # train_dataset = train_dataset.add_column("dataset_name", ["train"] * len(train_dataset))
    # test_dataset = test_dataset.add_column("dataset_name", ["test"] * len(test_dataset))

    encoder: SimpleEncoder = SimpleEncoder(
        model_name_or_path=args.model_name_or_path,
    )    
    
    # Create a list of sentences and a mapping from query_id to sentence and dataset_name
    sentences = train_dataset['query']
    query_ids = train_dataset['query_id']
    real_answers = train_dataset['answers']

    # Create a list of data points with their metadata
    data_points = []
    for query_id, sentence, single_answer in zip(query_ids, sentences, real_answers):
        data_points.append({
            'query_id': query_id,
            'sentence': sentence,
            'answers': single_answer
        })
    
    embeddings = []
    batch_size = 1000  # Adjust the batch size according to your GPU memory
    for i in tqdm.tqdm(range(0, len(sentences), batch_size), desc="Encoding sentences"):
        batch_sentences = sentences[i:i+batch_size]
        batch_embeddings = encoder.encode(batch_sentences)
        batch_embeddings = batch_embeddings.cpu().numpy()
        embeddings.append(batch_embeddings)
    embeddings = np.concatenate(embeddings, axis=0)
    
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    # Now perform clustering
    n_clusters = args.n_cluster  # Set the number of clusters to 10
    batch_size_kmeans = 10000  # You can adjust this if needed
    random_state = args.seed
    
    logger.info("Starting MiniBatchKMeans fitting...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size_kmeans, random_state=random_state)
    kmeans.fit(embeddings)
    
    # Assign clusters
    logger.info("Assigning clusters...")
    cluster_assignments = kmeans.predict(embeddings)
    
    # Map cluster index to list of data points
    cluster_to_data = defaultdict(list)
    cluster_to_train_sentence = defaultdict(list)
    for cluster_idx, data_point in zip(cluster_assignments, data_points):
        cluster_to_data[str(cluster_idx)].append(data_point['query_id'])
        cluster_to_train_sentence[str(cluster_idx)].append(data_point['answers'])

    # Save the mapping to a JSON file
    output_path = os.path.join(args.output_dir, f"{n_clusters}_train_cluster_to_data.json")
    with open(output_path, "w") as f:
        json.dump(cluster_to_data, f, ensure_ascii=False, indent=2)
    
    output_train_sentence_path = os.path.join(args.output_dir, f"{n_clusters}_train_sentence.json")
    with open(output_train_sentence_path, "w") as f:
        json.dump(cluster_to_train_sentence, f, ensure_ascii=False, indent=2)
    logger.info(f"Clustering complete. Output saved to: {output_path}")

    ############################################处理test
    test_dataset = load_dataset('json', data_files='{}/test.jsonl.gz'.format(args.data_dir), split='train')
    test_dataset = test_dataset.filter(lambda x: x['task_name'] in args.llm_eval_tasks[0].split())
    logger.info(f"Filtered Test Dataset length: {len(test_dataset)}")
    
    test_sentences = test_dataset['query']
    test_query_ids = test_dataset['query_id']
    test_real_answers = test_dataset['answers']

    test_data_points = []
    for query_id, sentence, test_single_answers in zip(test_query_ids, test_sentences, test_real_answers):
        test_data_points.append({
            'query_id': query_id,
            'sentence': sentence,
            'answers': test_single_answers
        })

    test_embeddings = []
    batch_size = 1000
    for i in tqdm.tqdm(range(0, len(test_sentences), batch_size), desc="Encoding test sentences"):
        batch_sentences = test_sentences[i:i+batch_size]
        batch_embeddings = encoder.encode(batch_sentences)
        batch_embeddings = batch_embeddings.cpu().numpy()
        test_embeddings.append(batch_embeddings)
    test_embeddings = np.concatenate(test_embeddings, axis=0)
    logger.info(f"Test Embeddings shape: {test_embeddings.shape}")

    # 使用已训练的 KMeans 模型对 test_embeddings 进行聚类分配
    logger.info("Assigning clusters to test dataset...")
    test_cluster_assignments = kmeans.predict(test_embeddings)

    # 将聚类分配结果映射到 query_id
    test_cluster_to_data = defaultdict(list)
    test_sentence = defaultdict(list)
    for cluster_idx, data_point in zip(test_cluster_assignments, test_data_points):
        test_cluster_to_data[str(cluster_idx)].append(data_point['query_id'])
        test_sentence[str(cluster_idx)].append(data_point['answers'])

    # 保存 test 聚类分配到 JSON 文件
    test_output_path = os.path.join(args.output_dir, f"{n_clusters}_test_cluster_to_data.json")
    with open(test_output_path, "w") as f:
        json.dump(test_cluster_to_data, f, ensure_ascii=False, indent=2)

    test_sentence_output_path = os.path.join(args.output_dir, f"{n_clusters}_test_sentence.json")
    with open(test_sentence_output_path, "w") as f:
        json.dump(test_sentence, f, ensure_ascii=False, indent=2)

    logger.info(f"Test clustering complete. Output saved to: {test_output_path}")

if __name__ == "__main__":
    main()
