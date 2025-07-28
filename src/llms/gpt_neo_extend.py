from logger_config import logger
from config import Arguments
from llms.gpt2_extend import GPT2_extend


# class GPTNeo_extend(GPT2_extend):

#     def __init__(self, args: Arguments, model_name_or_path: str = 'EleutherAI/gpt-neo-2.7B', **kwargs):
#         super().__init__(args, model_name_or_path, **kwargs)

class GPTNeo_extend(GPT2_extend):

    def __init__(self, args: Arguments, accelerator, model_name_or_path: str = 'EleutherAI/gpt-neo-2.7B', **kwargs):
        super().__init__(args=args, accelerator=accelerator, model_name_or_path=model_name_or_path, **kwargs)
        #GPT2_extend是基类，可以接受accelerator