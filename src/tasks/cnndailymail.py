from typing import Optional, List, Tuple, Dict, Union
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("cnndailymail")
class Cnndailymail(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'test'
        dataset = load_dataset('KaiLv/UDR_CNNDailyMail', split=split)
        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("{article}. TL;DR: ", "{highlights}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return None

    @property
    def metric_name(self) -> str:
        return 'generation'

    @property
    def task_name(self) -> str:
        return 'cnndailymail'

    def get_answer(self, example: Dict) -> Union[str, List[str]]:
        return example['highlights']
