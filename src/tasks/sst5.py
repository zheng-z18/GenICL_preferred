from typing import Optional, List, Tuple
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("sst5")
class Sst5(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'test'
        dataset = load_dataset("SetFit/sst5", split=split)
        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("Review: \"{text}\" Is this movie review text negative or positive?", "{answer}"),
            ("Short movie review: \"{text}\" Did the critic thinking positively or negatively of the movie?",
             "{answer}"),
            (
                "text from a movie review: \"{text}\" Was the movie seen positively or negatively based on the preceding review?",
                "{answer}"),
            ("\"{text}\" How would the sentiment of this text be perceived?", "{answer}"),
            ("Is the sentiment of the following text positive or negative? \"{text}\"", "{answer}"),
            ("What is the sentiment of the following movie review text? \"{text}\"", "{answer}"),
            ("{text}", "{answer}"),###
            ("{text}", "{answer}"),###
            ("Would the following phrase be considered positive or negative? \"{text}\"", "{answer}"),
            ("Does the following review have a positive or negative opinion of the movie? \"{text}\"", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        # return ["It was terrible", "It was bad", "It was okay", "It was good", "It was great"]
        return ["terrible", "bad", "okay", "good", "great"]

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'sst5'
