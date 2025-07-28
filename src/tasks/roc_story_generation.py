from typing import Optional, List, Tuple, Dict, Union
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("roc_story_generation")
class Roc_story_generation(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'test'
        dataset = load_dataset('KaiLv/UDR_RocStory', split=split)
        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("Beginning of the story: {question}. Rest of the story: ", "{target}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return None

    @property
    def metric_name(self) -> str:
        return 'generation'

    @property
    def task_name(self) -> str:
        return 'roc_story_generation'

    def get_answer(self, example: Dict) -> Union[str, List[str]]:
        return example['target']
