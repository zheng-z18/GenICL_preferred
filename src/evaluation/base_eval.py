from abc import abstractmethod
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from datasets import Dataset

from config import Arguments


class BaseEval:

    def __init__(self, args: Arguments, corpus: Dataset, **kwargs):
        self.args: Arguments = args
        # id / contents / task_name
        self.corpus: Dataset = corpus

        self.task_name_to_doc_ids: Dict[str, Set[str]] = defaultdict(set)
        for doc_id, task_name in zip(self.corpus['id'], self.corpus['task_name']):
            self.task_name_to_doc_ids[task_name].add(doc_id)# 将corpus直接按task划分，存为字典

    @abstractmethod
    def get_topk_score_doc_ids(
            self, queries: List[str], k: int, task_names: List[str]
    ) -> List[List[Tuple[float, str]]]:
        raise NotImplementedError

    def get_doc_ids_by_task_name(self, task_name: str) -> List[str]:
        return list(self.task_name_to_doc_ids[task_name])

    # def get_prompt_by_doc_ids(self, doc_ids: List[str]) -> str:
    #     return '\n\n'.join([self.corpus[int(doc_id)]['contents'] for doc_id in doc_ids])

    def get_prompt_by_doc_ids(self, doc_ids: List[str]) -> str:
        prompts = []
        for doc_id in doc_ids:
            corpus_entry = self.corpus[int(doc_id)]
            if int(corpus_entry['id']) != int(doc_id):# 验证 doc_id 是否与 corpus 中的 id 对应
                raise ValueError(f"doc_id {doc_id} does not match corpus entry id {corpus_entry['id']}")
            prompts.append(corpus_entry['contents'])
        return '\n\n'.join(prompts) ##\n\n\n
    
    def channel_get_prompt_by_doc_ids(self, doc_ids: List[str]) -> str:
        prompts = []
        for doc_id in doc_ids:
            corpus_entry = self.corpus[int(doc_id)]
            if int(corpus_entry['id']) != int(doc_id):# 验证 doc_id 是否与 corpus 中的 id 对应
                raise ValueError(f"doc_id {doc_id} does not match corpus entry id {corpus_entry['id']}")
            content = corpus_entry['contents']
            assert content.count('\n') == 1
            before_newline, after_newline = content.split('\n', 1)
            content = after_newline + '\n' + before_newline
            prompts.append(content)
            # prompts.append(corpus_entry['contents'])
        return '\n\n'.join(prompts) #\n\n\n

    def channel_get_prompt_by_train(self, doc_ids: List[str], task_corpus) -> str:
        prompts = []
        for doc_id in doc_ids:
            # corpus_entry = task_corpus
            corpus_entry = task_corpus.filter(lambda x: x['id'] == doc_id)
            # corpus_entry = self.corpus[int(doc_id)]
            # if int(corpus_entry['id']) != int(doc_id):# 验证 doc_id 是否与 corpus 中的 id 对应
            #     raise ValueError(f"doc_id {doc_id} does not match corpus entry id {corpus_entry['id']}")
            content = corpus_entry['contents'][0]
            assert content.count('\n') == 1
            before_newline, after_newline = content.split('\n', 1)
            content = after_newline + '\n' + before_newline
            prompts.append(content)
            # prompts.append(corpus_entry['contents'])
        return '\n\n\n'.join(prompts)
