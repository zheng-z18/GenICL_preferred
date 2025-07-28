import os

from datasets import Dataset

from llms import BaseLLM, GPT2, GPTNeo, Llama,GPT2_MT, GPTNeo_MT, Llama_MT, GPT2_extend, Llama_extend, GPT2_multi_theta, GPTNeo_extend, GPT2_parallel, GPTNeo_parallel, Llama_parallel, GPT2_probability, Llama_probability
from evaluation import BaseEval, RandomEval, DenseEval
from config import Arguments
from logger_config import logger

def build_multi_theta_llm(args: Arguments) -> BaseLLM:
    model_name_or_path: str = args.llm_model_name_or_path
    if 'gpt2' in model_name_or_path:
        if args.llm_max_input_length >= 1024:
            args.llm_max_input_length -= max(args.llm_max_decode_length, 128)
            logger.warning('GPT2 models cannot handle sequences longer than 1024. '
                           'set to {}'.format(args.llm_max_input_length))
        llm = GPT2_multi_theta(args=args, model_name_or_path=model_name_or_path)
    elif 'gpt-neo' in model_name_or_path:
        llm = GPTNeo(args=args, model_name_or_path=model_name_or_path)
    elif 'llama' in model_name_or_path:
        llm = Llama_extend(args=args, model_name_or_path=model_name_or_path)
    else:
        raise ValueError('Invalid model name or path: {}'.format(model_name_or_path))

    return llm


def build_extend_llm(args: Arguments, accelerator) -> BaseLLM:
    model_name_or_path: str = args.llm_model_name_or_path
    if 'gpt2' in model_name_or_path:
        if args.llm_max_input_length >= 1024:
            args.llm_max_input_length -= max(args.llm_max_decode_length, 128)
            logger.warning('GPT2 models cannot handle sequences longer than 1024. '
                           'set to {}'.format(args.llm_max_input_length))
        llm = GPT2_extend(args=args, accelerator=accelerator, model_name_or_path=model_name_or_path)
    elif 'gpt-neo' in model_name_or_path:
        llm = GPTNeo_extend(args=args, accelerator=accelerator, model_name_or_path=model_name_or_path)
    elif 'llama' in model_name_or_path:
        llm = Llama_extend(args=args, accelerator=accelerator, model_name_or_path=model_name_or_path)
    else:
        raise ValueError('Invalid model name or path: {}'.format(model_name_or_path))

    return llm

def build_MT_llm(args: Arguments, accelerator) -> BaseLLM:
    model_name_or_path: str = args.llm_model_name_or_path
    if 'gpt2' in model_name_or_path:
        if args.llm_max_input_length >= 1024:
            args.llm_max_input_length -= max(args.llm_max_decode_length, 128)
            logger.warning('GPT2 models cannot handle sequences longer than 1024. '
                           'set to {}'.format(args.llm_max_input_length))
        llm = GPT2_MT(args=args, accelerator=accelerator, model_name_or_path=model_name_or_path)
    elif 'gpt-neo' in model_name_or_path:
        llm = GPTNeo_MT(args=args, accelerator=accelerator, model_name_or_path=model_name_or_path)
    elif 'llama' in model_name_or_path:
        llm = Llama_MT(args=args, accelerator=accelerator, model_name_or_path=model_name_or_path)
    else:
        raise ValueError('Invalid model name or path: {}'.format(model_name_or_path))

    return llm

def build_llm_parallel(args: Arguments, accelerator) -> BaseLLM:
    model_name_or_path: str = args.llm_model_name_or_path
    if 'gpt2' in model_name_or_path:
        if args.llm_max_input_length >= 1024:
            args.llm_max_input_length -= max(args.llm_max_decode_length, 128)
            logger.warning('GPT2 models cannot handle sequences longer than 1024. '
                           'set to {}'.format(args.llm_max_input_length))
        llm = GPT2_parallel(args=args, accelerator=accelerator, model_name_or_path=model_name_or_path)
    elif 'gpt-neo' in model_name_or_path:
        llm = GPTNeo_parallel(args=args, accelerator=accelerator, model_name_or_path=model_name_or_path)
    elif 'llama' in model_name_or_path:
        llm = Llama_parallel(args=args, accelerator=accelerator, model_name_or_path=model_name_or_path)
    else:
        raise ValueError('Invalid model name or path: {}'.format(model_name_or_path))
    
    return llm

def build_llm_probability(args: Arguments) -> BaseLLM:
    model_name_or_path: str = args.llm_model_name_or_path
    if 'gpt2' in model_name_or_path:
        if args.llm_max_input_length >= 1024:
            args.llm_max_input_length -= max(args.llm_max_decode_length, 128)
            logger.warning('GPT2 models cannot handle sequences longer than 1024. '
                           'set to {}'.format(args.llm_max_input_length))
        llm = GPT2_probability(args=args, model_name_or_path=model_name_or_path)
    # elif 'gpt-neo' in model_name_or_path:
    #     llm = GPTNeo_probability(args=args, accelerator=accelerator, model_name_or_path=model_name_or_path)
    elif 'llama' in model_name_or_path:
        llm = Llama_probability(args=args, model_name_or_path=model_name_or_path)
    else:
        raise ValueError('Invalid model name or path: {}'.format(model_name_or_path))
    
    return llm

def build_llm(args: Arguments) -> BaseLLM:
    model_name_or_path: str = args.llm_model_name_or_path
    if 'gpt2' in model_name_or_path:
        if args.llm_max_input_length >= 1024:
            args.llm_max_input_length -= max(args.llm_max_decode_length, 128)
            logger.warning('GPT2 models cannot handle sequences longer than 1024. '
                           'set to {}'.format(args.llm_max_input_length))
        llm = GPT2(args=args, model_name_or_path=model_name_or_path)
    elif 'gpt-neo' in model_name_or_path:
        llm = GPTNeo(args=args, model_name_or_path=model_name_or_path)
    elif 'llama' in model_name_or_path:
        llm = Llama(args=args, model_name_or_path=model_name_or_path)
    else:
        raise ValueError('Invalid model name or path: {}'.format(model_name_or_path))

    return llm


def build_eval_model(args: Arguments, corpus: Dataset) -> BaseEval:
    model_name_or_path: str = args.model_name_or_path
    if model_name_or_path == 'random':
        return RandomEval(args=args, corpus=corpus)
    else:
        return DenseEval(args=args, corpus=corpus)


def parse_model_id(model_name_or_path: str) -> str:
    return os.path.basename(model_name_or_path.strip('/'))[-12:]
