from typing import List, Dict, Tuple
from datasets import Dataset

from evaluation.base_eval import BaseEval
from config import Arguments
from logger_config import logger


class DenseEval(BaseEval):

    def __init__(self, args: Arguments, corpus: Dataset, **kwargs):
        super().__init__(args, corpus, **kwargs)

        input_prefix = 'query: ' if args.add_qd_prompt else ''
        # TODO: Hack
        is_e5_model = any(e5_name in args.model_name_or_path for e5_name in ['intfloat/e5', 'intfloat/multilingual-e5', 'model/e5-base', 'e5-base'])
        if is_e5_model and not input_prefix:
            logger.warning('E5 models need input prefix, set input_prefix = "query: "')
            input_prefix = 'query: '

        from models import SimpleEncoder, SimpleRetriever
        encoder: SimpleEncoder = SimpleEncoder( #小模型e5-base 
            model_name_or_path=args.model_name_or_path,
            l2_normalize=args.l2_normalize,
            prompt=input_prefix,
        )
        cache_dir = '{}/embeddings/'.format(args.output_dir)

        self.retriever: SimpleRetriever = SimpleRetriever(
            encoder=encoder,
            corpus=corpus,
            cache_dir=cache_dir,
        )

    def get_topk_score_doc_ids(self, queries: List[str], k: int, task_names: List[str]) -> List[List[Tuple[float, str]]]:
        assert len(queries) == len(task_names)

        query_idx_to_topk: Dict[int, List[Tuple]] = self.retriever.search_topk(queries=queries, top_k=k) ##每个query检索出k个
        for idx in range(len(queries)):
            q_task_name = task_names[idx]
            for j, (score, doc_id) in enumerate(query_idx_to_topk[idx]):#score是样本得分，doc_id是样本在整个数据集的索引
                if str(doc_id) not in self.task_name_to_doc_ids[q_task_name]:#样本不在query的任务对应的样本集合和中，即这个样本和query不是一个任务
                    query_idx_to_topk[idx][j] = (score - 100., doc_id)
                    # query_idx_to_topk[idx][j] = (score, doc_id)
            query_idx_to_topk[idx] = sorted(query_idx_to_topk[idx], key=lambda x: x[0], reverse=True) #x[0]指定按照第一个元素即分数 排序，reverse=True表示降序排列

        topk_score_doc_ids: List[List[Tuple[float, str]]] = []
        for idx in range(len(queries)):
            score_doc_ids: List[Tuple[float, str]] = query_idx_to_topk[idx][:k] ##每个query(idx表示query索引)的前k个相关样本
            score_doc_ids = [(score, str(doc_id)) for score, doc_id in score_doc_ids] #类型转换
            topk_score_doc_ids.append(score_doc_ids)

        return topk_score_doc_ids

