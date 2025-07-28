import torch

from dataclasses import dataclass
from transformers import BatchEncoding, DataCollatorWithPadding, PreTrainedTokenizerBase
from typing import List, Dict, Any, Union, Optional
from transformers.file_utils import PaddingStrategy

@dataclass
class DPOCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    delimiter: str = '\n'
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [f['prompt'] for f in features]
        chosens = [f['chosen'] for f in features]
        rejecteds = [f['rejected'] for f in features]

        # 对 'prompt' + 'chosen' 进行编码
        chosen_encodings = self.tokenizer(
            prompts,
            chosens,
            truncation=True,
            max_length=self.max_length,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt'
        )

        # 对 'prompt' + 'rejected' 进行编码
        rejected_encodings = self.tokenizer(
            prompts,
            rejecteds,
            truncation=True,
            max_length=self.max_length,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt'
        )

        batch = {
            'chosen_input_ids': chosen_encodings['input_ids'],
            'chosen_attention_mask': chosen_encodings['attention_mask'],
            'rejected_input_ids': rejected_encodings['input_ids'],
            'rejected_attention_mask': rejected_encodings['attention_mask'],
        }

        return batch

@dataclass
class CrossEncoderCollator(DataCollatorWithPadding):

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        unpack_features = []
        for ex in features:
            keys = list(ex.keys())
            for idx in range(len(ex[keys[0]])):
                unpack_features.append({k: ex[k][idx] for k in keys}) #对输入数据进行解包

        collated_batch_dict = self.tokenizer.pad(
            unpack_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors)

        collated_batch_dict['labels'] = torch.zeros(len(features), dtype=torch.long) #为collated_batch_dict添加lebels字段
        return collated_batch_dict

@dataclass
class Latent2Collator(DataCollatorWithPadding):

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        unpack_features = []
        for ex in features:
            keys = list(ex.keys())
            for idx in range(len(ex[keys[0]])):
                unpack_features.append({k: ex[k][idx] for k in keys}) #对输入数据进行解包

        collated_batch_dict = self.tokenizer.pad(
            unpack_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors)
        
        return collated_batch_dict
