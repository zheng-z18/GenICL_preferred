from typing import Optional, List, Tuple
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("subj")
class Subj(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'test' #validation
        dataset = load_dataset('SetFit/subj', split=split)
        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("Review: \"{text}\" Is this movie review sentence negative or positive?", "{answer}"),
            ("Short movie review: \"{sentence}\" Did the critic thinking positively or negatively of the movie?",
             "{answer}"),
            (
                "Sentence from a movie review: \"{sentence}\" Was the movie seen positively or negatively based on the preceding review?",
                "{answer}"),
            ("\"{sentence}\" How would the sentiment of this sentence be perceived?", "{answer}"),
            ("Is the sentiment of the following sentence positive or negative? \"{sentence}\"", "{answer}"),
            ("What is the sentiment of the following movie review sentence? \"{sentence}\"", "{answer}"),
            ("{sentence}", "{answer}"),###
            ("{text}", "{answer}"),###
            ("Would the following phrase be considered positive or negative? \"{sentence}\"", "{answer}"),
            ("Does the following review have a positive or negative opinion of the movie? \"{sentence}\"", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        # return ['This is objective', 'This is subjective']
        return ['objective', 'subjective']

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'subj'
