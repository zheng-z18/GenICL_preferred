import logging
import os
from transformers.utils.logging import enable_explicit_format, set_verbosity_info, set_verbosity_warning
from transformers.trainer_callback import PrinterCallback
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    set_seed,
    PreTrainedModel,
    AutoModelForCausalLM
)
from logger_config import logger, LoggerCallback
from config import Arguments
from loaders import KTODataset_model
from trl import KTOConfig, KTOTrainer
from peft import PromptTuningConfig, PrefixTuningConfig, get_peft_model, PeftModel, LoraConfig
from datasets import Dataset, load_dataset, DownloadMode, concatenate_datasets
from tqdm import tqdm
from typing import List, Tuple, Any, Optional, Union, Dict
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import warnings
from utils import save_dataset
from inference.inference_utils import get_prompt_save_path


# import debugpy
# try:
#     debugpy.listen(("localhost", 9609))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass



def _common_setup(args: Arguments):
    set_verbosity_info()
    if args.process_index > 0:
        logger.setLevel(logging.WARNING)
        set_verbosity_warning()
    enable_explicit_format()
    set_seed(args.seed)

class CustomKTOTrainer(KTOTrainer):
    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        """Compute the KTO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        batch = {k: (v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        forward_output = self.forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_KL_logps,
        ) = forward_output[:5]
        constant_value = 1.0
        all_num = 1
        metrics["rewards/chosen_sum"] = constant_value
        metrics["logps/chosen_sum"] = constant_value
        metrics["logits/chosen_sum"] = constant_value
        metrics["count/chosen"] = all_num
        metrics["rewards/rejected_sum"] = constant_value
        metrics["logps/rejected_sum"] = constant_value
        metrics["logits/rejected_sum"] = constant_value
        metrics["count/rejected"] = all_num


        gathered_policy_chosen_logps = self.accelerator.gather(policy_chosen_logps).detach().cpu().tolist()
        metrics["policy_chosen_logps"] = gathered_policy_chosen_logps####################################
        loss = torch.tensor(1.0, requires_grad=False, device=self.accelerator.device)

        return loss, metrics
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # train metrics should have no prefix, eval should have 'eval_'
        prefix = "eval_" if train_eval == "eval" else ""
        # accumulate average metrics from sums and lengths
        for split in ["chosen", "rejected"]:
            if f"count/{split}" in self._stored_metrics[train_eval]:
                count_sum = torch.Tensor(self._stored_metrics[train_eval][f"count/{split}"]).sum().item()
                for metric in ["rewards", "logps", "logits"]:
                    logs[f"{prefix}{metric}/{split}"] = (
                        torch.Tensor(self._stored_metrics[train_eval][f"{metric}/{split}_sum"]).sum().item()
                        / count_sum
                    )
                    # delete obsolete metric
                    del self._stored_metrics[train_eval][f"{metric}/{split}_sum"]
                del self._stored_metrics[train_eval][f"count/{split}"]
        # calculate reward margin
        if f"{prefix}rewards/chosen" in logs and f"{prefix}rewards/rejected" in logs:
            logs[f"{prefix}rewards/margins"] = logs[f"{prefix}rewards/chosen"] - logs[f"{prefix}rewards/rejected"]
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():#####################
            if key == 'kl':
                logs[f"{prefix}{key}"] = torch.Tensor(metrics).mean().item()
            else:
                logs[f"{prefix}{key}"] = [item for sublist in metrics for item in sublist]
        del self._stored_metrics[train_eval]
        return super().log(logs)


    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        else:
            # Fixes end-dec RuntimeError
            labels = labels.clone()

        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0 #label_pad_token_id -100

        # per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2) #[2,99]  对 logits 的最后一维（vocab_size 维度）进行 softmax 运算并取对数，计算每个 token 的对数概率。
        logits_log_softmax = logits.log_softmax(-1)
        expanded_labels = labels.unsqueeze(2)
        per_token_logps_1 = torch.gather(logits_log_softmax, dim=2, index=expanded_labels)
        per_token_logps = per_token_logps_1.squeeze(2)

        avg_log_probs = per_token_logps * loss_mask

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1) #每个序列总的对数概率，


def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    _common_setup(args)
    logger.info('Args={}'.format(str(args)))

    file_out_path: str = get_prompt_save_path(args=args)
    if os.path.exists(file_out_path):
        logger.info('Prompt file {} exists. Skip.'.format(file_out_path))
        return


    args.llm_eval_tasks = sorted(args.llm_eval_tasks)
    ########################################################tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    #########################################################model
    model = AutoModelForCausalLM.from_pretrained(args.llm_model_name_or_path)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, args.prefix_embed_file)

    # assert len(tokenizer) == model.config.vocab_size, f"Vocab sizes do not match: {len(tokenizer)} != {model.config.vocab_size}"

    #########################################################dataset

    corpus: Dataset = load_dataset(
        'json', data_files=os.path.join(args.data_dir,'passages.jsonl.gz'), split='train')
    eval_dataset: Dataset = load_dataset(
        'json', data_files=os.path.join(args.data_dir,'e5-base-v2_test.jsonl.gz'), split='train')
    task_ds: Dataset = eval_dataset.filter(lambda x: x['task_name'] in args.llm_eval_tasks)

    if len(task_ds) > 1000:
        task_ds = task_ds.shuffle(seed=args.seed).select(range(1000))

    test_dataset_json = []
    docs_num = 100 ###########每个query跑多少个doc
    all_doc_ids = [] 
    for each_data in tqdm(task_ds, desc="Processing data", unit="item"):
        query = each_data['query']
        assert len(each_data['doc_ids']) >= docs_num
        doc_ids = each_data['doc_ids'][:docs_num] 
        # doc_ids = each_data['doc_ids'][:docs_num] if len(each_data['doc_ids']) >= docs_num else each_data['doc_ids']
        all_doc_ids.append(doc_ids)
        task_name = each_data['task_name']
        theta_description = ''.join([
                f'<{task_name}-{idx}>' 
                for idx in range(args.n_prefix_tokens)
            ]) 
        for doc_id in doc_ids:
            content = corpus[int(doc_id)]['contents']
            demo_and_x = content + '\n' + query
            test_dataset_json.append({
            'prompt': demo_and_x,
            'completion': theta_description,
        })
        # break    ###################################
    
    data_dict = {
        'prompt': [item['prompt'] for item in test_dataset_json],
        'completion': [item['completion'] for item in test_dataset_json],
        'label': [True for _ in test_dataset_json]
    }

    test_dataset = Dataset.from_dict(data_dict)

    
    ########################################################trainer
    kto_config = KTOConfig(
        output_dir = args.output_dir,
        do_eval=True,
        do_predict=True,
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size = args.per_device_eval_batch_size,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        logging_dir="log/",
        logging_strategy="no",
        report_to=None,
        gradient_checkpointing = args.gradient_checkpointing,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        fp16 = args.fp16,
        max_prompt_length=128, #256
        max_completion_length=128, #32
    )

    dummy_train_dataset = Dataset.from_dict({
        "prompt": [],
        "completion": [],
        "label": [],
    })

    trainer: Trainer = CustomKTOTrainer( #KTOTrainer
        model=model,
        ref_model=None,
        args=kto_config,
        train_dataset=dummy_train_dataset, #test_dataset
        eval_dataset=test_dataset, #test_dataset
        tokenizer=tokenizer,
    )

    predictions = trainer.evaluate()
    all_chosen_logits = predictions["eval_policy_chosen_logps"]
    # all_chosen_rewards = predictions["eval_policy_chosen_rewards"]

    print(f'len(all_chosen_logits):{len(all_chosen_logits)}')
    print(f'len(test_dataset):{len(test_dataset)}')
    assert len(all_chosen_logits) == len(test_dataset), ("Length mismatch")

    topk_doc_ids = []
    topk_scores = []
    start_idx = 0 # 所以可以用一个游标 start 来切片取出当前 query 的那 docs_num 条打分
    for i, each_data in enumerate(task_ds):
        query = each_data["query"]
        current_doc_ids = all_doc_ids[i]  # 长度应是 docs_num
        assert set(current_doc_ids).issubset(set(each_data["doc_ids"]))
        assert len(current_doc_ids) == docs_num
        slice_logits = all_chosen_logits[start_idx : start_idx + docs_num] ## 截取本 query 对应的 logits 切片
        # slice_logits[j] 就是 current_doc_ids[j] 的得分

        # 为了得到 top-k：
        # 1) 对 slice_logits 做排序，索引按分数从大到小排列
        sorted_indices = sorted(range(len(slice_logits)),key=lambda idx: slice_logits[idx],reverse=True)
        # 2) 取前 k 个索引
        topk_indices = sorted_indices[:args.llm_k_shot]
        # 3) 映射到 doc_id
        single_topk_doc_ids = [current_doc_ids[idx] for idx in topk_indices]
        # 4) 也可以拿到对应的 logits
        single_topk_scores = [slice_logits[idx] for idx in topk_indices]
        
        topk_doc_ids.append(single_topk_doc_ids)
        topk_scores.append(single_topk_scores)

        start_idx += docs_num ## 游标向后移动
    
    
    def get_prompt_by_doc_ids(doc_ids: List[str]) -> str:
        prompts = []
        for doc_id in doc_ids:
            corpus_entry = corpus[int(doc_id)]
            if int(corpus_entry['id']) != int(doc_id):# 验证 doc_id 是否与 corpus 中的 id 对应
                raise ValueError(f"doc_id {doc_id} does not match corpus entry id {corpus_entry['id']}")
            prompts.append(corpus_entry['contents'])
        return '\n\n'.join(prompts)
    input_prompts: List[str] = [get_prompt_by_doc_ids(doc_ids) for doc_ids in topk_doc_ids]
    task_ds = task_ds.add_column('input_prompt', input_prompts)
    task_ds = task_ds.add_column('topk_doc_ids', topk_doc_ids)
    task_ds = task_ds.add_column('topk_scores', topk_scores)

    if trainer.is_world_process_zero():
        save_dataset(task_ds, file_out_path)
        logger.info('Save {} examples to {}'.format(len(task_ds), file_out_path))
    return

if __name__ == "__main__":
    main()
