from datasets import Dataset, load_dataset, DownloadMode
from transformers import HfArgumentParser

from config import Arguments
from logger_config import logger
from llms import BaseLLM
from model_utils import build_llm
from data_utils import log_task_statistics
from llm_evaluator import LLMEvaluator
from inference.inference_utils import get_prompt_save_path

# import debugpy
# try:
#     debugpy.listen(("localhost", 9609))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]


def main():
    # columns: query_id / query / answers / task_name / input_prompt
    eval_dataset: Dataset = load_dataset(
        'json', data_files='{}/{}_k{}.jsonl.gz'.format(args.output_dir, args.llm_eval_split, args.llm_k_shot), split='train') #download_mode='force_redownload'
    if not args.llm_eval_tasks or args.llm_eval_tasks[0] == 'all':
        args.llm_eval_tasks = sorted(eval_dataset.unique('task_name'))#30个任务
        logger.info('Eval all {} tasks'.format(len(args.llm_eval_tasks)))

    if args.process_index <= 0:
        log_task_statistics(eval_dataset, split=args.llm_eval_split)
        logger.info('{} tasks to evaluate: {}'.format(len(args.llm_eval_tasks), args.llm_eval_tasks))
    

    llm: BaseLLM = build_llm(args=args)
    llm.cuda(args.process_index)

    evaluator: LLMEvaluator = LLMEvaluator(args=args, llm=llm)

    for task_name in args.llm_eval_tasks:
        logger.info('Evaluating task: {}'.format(task_name))
        evaluator.eval_single_task(eval_dataset, task_name)
        if args.dry_run:
            break

    logger.info('Done')


if __name__ == '__main__':
    main()
