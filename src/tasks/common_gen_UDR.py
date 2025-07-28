from typing import Optional, List, Tuple, Dict, Union
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("common_gen")
class Common_gen(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'validation'  #test没有target，无法使用
        # dataset = load_dataset('KaiLv/UDR_CommonGen', split=split)
        dataset = load_dataset('allenai/common_gen', split=split)
        dataset = dataset.map(lambda ex: {'concepts': ", ".join(ex["concepts"])}) #将[ "field", "look", "stand" ]转为field, look, stand。方便下面template直接使用
        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("Generate a sentence using these concepts: {concepts}. Generated sentence: ", "{target}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return None

    @property
    def metric_name(self) -> str:
        return 'generation'

    @property
    def task_name(self) -> str:
        return 'common_gen'

    def get_answer(self, example: Dict) -> Union[str, List[str]]:
        return example['target']
