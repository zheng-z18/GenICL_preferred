import logging
import os
from transformers.utils.logging import enable_explicit_format, set_verbosity_info, set_verbosity_warning
from transformers.trainer_callback import PrinterCallback
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    set_seed,
    AutoModelForCausalLM
)
from logger_config import logger, LoggerCallback
from config import Arguments
from loaders import KTODataset_theta_lora
from trl import KTOConfig, KTOTrainer
from peft import PromptTuningConfig, PrefixTuningConfig, get_peft_model, PeftModel, LoraConfig

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


def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    _common_setup(args)
    logger.info('Args={}'.format(str(args)))


    args.llm_eval_tasks = sorted(args.llm_eval_tasks)
    ########################################################tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    #########################################################model
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.config.use_cache = False

    # assert len(tokenizer) == model.config.vocab_size, f"Vocab sizes do not match: {len(tokenizer)} != {model.config.vocab_size}"

    peft_config = LoraConfig(
        r=32, #64
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,  #64
        lora_dropout=0,  # 设置为0以优化性能
        bias="none",  # 设置为"none"以优化性能
    )

    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()
    # return

    #########################################################dataset
    train_dataset = KTODataset_theta_lora(args=args, split='train')
    train_dataset.save_dataset(os.path.join(args.output_dir, "train_datasets.json"))
    validation_dataset = KTODataset_theta_lora(args=args, split='validation')
    validation_dataset.save_dataset(os.path.join(args.output_dir, "val_datasets.json"))
    
    ########################################################trainer
    kto_config = KTOConfig(
        output_dir = args.output_dir,
        learning_rate = args.learning_rate,
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size = args.per_device_eval_batch_size,
        max_steps = args.max_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing = args.gradient_checkpointing,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        # weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        save_only_model = True, #True只保存embedding layer, trl 0.9.3不支持该项目
        fp16 = args.fp16,
        logging_dir=args.logging_dir,
        report_to="tensorboard",
        max_prompt_length=128, #256
        max_completion_length=128, #32
        desirable_weight=1,
        undesirable_weight=1
    )

    trainer: Trainer = KTOTrainer(
        model=model,
        ref_model=None,
        args=kto_config,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LoggerCallback)
    
    # initial_checkpoint_dir = os.path.join(args.output_dir, "checkpoint-0")
    # trainer.save_model(output_dir=initial_checkpoint_dir)

    if args.do_train:
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    return


if __name__ == "__main__":
    main()
