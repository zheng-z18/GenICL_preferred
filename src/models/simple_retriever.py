import os
import tqdm
import torch

from typing import List, Dict, Union
from datasets import Dataset
from collections import defaultdict

from models.simple_encoder import SimpleEncoder
from logger_config import logger


def _sharded_search_topk(
        query_embeds: torch.Tensor, top_k: int,  #18*768
        shard_embed: torch.Tensor, shard_idx: int,
        idx_offset: int) -> Dict[int, List]:
    query_idx_to_topk: Dict[int, List] = defaultdict(list)
    search_batch_size = 256
    query_indices = list(range(query_embeds.shape[0])) #长度为591391，和query对应，是query的索引

    for start in tqdm.tqdm(range(0, query_embeds.shape[0], search_batch_size),#search_batch_size是步长
                           desc="search shard {}".format(shard_idx),
                           mininterval=5):
        batch_query_embed = query_embeds[start:(start + search_batch_size)]
        batch_query_indices = query_indices[start:(start + search_batch_size)]##每次拿一个batch的query和对应的索引
        batch_score = torch.mm(batch_query_embed, shard_embed.t())  ###torch.mm 两个矩阵乘法，query和分片shard的相似性
        batch_sorted_score, batch_sorted_indices = torch.topk(batch_score, k=top_k, dim=-1, largest=True)
        for batch_idx, query_idx in enumerate(batch_query_indices):#遍历每个query
            cur_scores = batch_sorted_score[batch_idx].cpu().tolist() #取batch_sorted_score的第一个维度，表示一个query对应的分区shard的各个的相似度
            cur_indices = [str(idx + idx_offset) for idx in batch_sorted_indices[batch_idx].cpu().tolist()]#batch_sorted_indices按分数排序后的索引   + idx_offset在整个数据集中的真实索引
            query_idx_to_topk[query_idx] += list(zip(cur_scores, cur_indices)) #得分和索引放一起
            query_idx_to_topk[query_idx] = sorted(query_idx_to_topk[query_idx], key=lambda t: -t[0])[:top_k]  #前k个最高分数的结果

    return query_idx_to_topk #591391行，每行表示每个query的前100个相关


class SimpleRetriever:
    def __init__(self, encoder: SimpleEncoder,
                 corpus: Union[Dataset, List[str]],
                 cache_dir: str = None):
        self.encoder = encoder

        # Encode the "contents" column of the corpus
        if isinstance(corpus, List):
            corpus = Dataset.from_dict({'contents': corpus})
        self.corpus: Dataset = corpus
        logger.info(f"Corpus size: {len(self.corpus)}")

        self.cache_dir = cache_dir or 'tmp-{}/'.format(len(corpus))
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Cache dir: {self.cache_dir}")
        self.encode_shard_size = 2_000_000

    def search_topk(self, queries: List[str], top_k: int = 10) -> Dict[int, List]:
        # encode the corpus or load from cache if it already exists
        self._encode_corpus_if_necessary(shard_size=self.encode_shard_size)

        # encode the queries
        query_embeds = self._encode_queries(queries)  #591391*768
        if torch.cuda.is_available():
            query_embeds = query_embeds.cuda()

        # search the top-k results
        query_idx_to_topk: Dict[int, List] = defaultdict(list)
        num_shards = (len(self.corpus) + self.encode_shard_size - 1) // self.encode_shard_size
        idx_offset = 0
        
        for shard_idx in range(num_shards):#query_embeds分别在不同的分片shard中找最相关的top-k个，不断更新
            out_path: str = self._get_out_path(shard_idx)
            shard_embeds = torch.load(out_path, map_location=lambda storage, loc: storage) #1570780*768
            shard_embeds = shard_embeds.to(query_embeds.device)
            shard_query_idx_to_topk = _sharded_search_topk(  #每行对应一个query，包含100个最相关样本的得分和在整体数据集的索引
                query_embeds=query_embeds,
                top_k=top_k,
                shard_embed=shard_embeds,
                shard_idx=shard_idx, 
                idx_offset=idx_offset
            )
            for query_idx, shard_topk in shard_query_idx_to_topk.items():# 遍历每行的query query_idx：第几个query；shard_topk：前100个样本（得分，索引）
                query_idx_to_topk[query_idx] += shard_topk #将这个分片的100个加到原来的100个后面，下一行在一块挑出新的100个
                query_idx_to_topk[query_idx] = sorted(query_idx_to_topk[query_idx], key=lambda t: -t[0])[:top_k]

            idx_offset += shard_embeds.shape[0]

        return query_idx_to_topk  #591391行query，每行对应100个最相关样本（得分，索引）

    def encode_corpus(self):
        self._encode_corpus_if_necessary(shard_size=self.encode_shard_size)
        logger.info('Done encoding corpus')

    def _get_out_path(self, shard_idx: int) -> str:
        return '{}/shard_{}'.format(self.cache_dir, shard_idx)

    def _encode_corpus_if_necessary(self, shard_size: int):#shard_size 每个分片的数据量
        num_shards = (len(self.corpus) + shard_size - 1) // shard_size
        num_examples = 0
        for shard_idx in range(num_shards):
            out_path: str = self._get_out_path(shard_idx)
            if os.path.exists(out_path):
                logger.info('{} already exists, will skip encoding'.format(out_path))
                num_examples += len(torch.load(out_path, map_location=lambda storage, loc: storage))
                continue
            shard_dataset: Dataset = self.corpus.shard(  #创建当前分片的数据集
                num_shards=num_shards,#总的分片数
                index=shard_idx,#当前分片索引
                contiguous=True#分片是连续的
            )
            shard_embeds: torch.Tensor = self.encoder.encode(
                sentences=shard_dataset['contents']
            )

            num_examples += shard_embeds.shape[0]
            logger.info('Saving shard {} ({} examples) to {}'.format(shard_idx, len(shard_dataset), out_path))
            torch.save(shard_embeds, out_path) #将编码后的保存到out_path

        assert num_examples == len(self.corpus), \
            f"Number of examples in the corpus ({len(self.corpus)}) " \
            f"does not match the number of examples in the shards ({num_examples})"  #确保处理的=原来的，确保没丢数据

    def _encode_queries(self, queries: List[str]) -> torch.Tensor:
        return self.encoder.encode(
            sentences=queries
        )
