from logger_config import logger
from config import Arguments
from llms.gpt2 import GPT2
from llms.gpt2_extend import GPT2_extend


class Llama_extend(GPT2_extend):

    # def __init__(self, args: Arguments, model_name_or_path: str = 'huggyllama/llama-7b', **kwargs):
    #     super().__init__(args, model_name_or_path, **kwargs)
    def __init__(self, args: Arguments, accelerator, model_name_or_path: str = 'huggyllama/llama-7b', **kwargs):
        super().__init__(args=args, accelerator=accelerator, model_name_or_path=model_name_or_path, **kwargs)