import os
import torch
from typing import Optional
from transformers.trainer import Trainer, IntervalStrategy
from transformers.modeling_outputs import SequenceClassifierOutput

from logger_config import logger
from evaluation.metrics import accuracy
from utils import AverageMeter


class Latent2Trainer(Trainer):

    def __init__(self, *pargs, **kwargs):
        super(Latent2Trainer, self).__init__(*pargs, **kwargs)

        self.acc_meter = AverageMeter('acc', round_digits=2)
        self.last_epoch = 0

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to {}".format(output_dir))

        # self.model.save_pretrained(output_dir)

        embedding_path = os.path.join(output_dir, "embedding_layer.pth")
        embedding_layer = self.model.hf_model.get_input_embeddings()
        torch.save(embedding_layer, embedding_path)

        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

    def compute_loss(self, model, inputs, return_outputs=False):
        # outputs = model(inputs)
        # loss = outputs.loss
        loss = model(inputs)
        # outputs = loss
        if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            logger.info(f"Step: {self.state.global_step}")
        return loss
        # return (loss, outputs) if return_outputs else loss
        # if self.model.training:
        #     labels = inputs['labels']
        #     step_acc = accuracy(output=outputs.logits.detach(), target=labels)[0]
        #     self.acc_meter.update(step_acc)
        #     if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
        #         logger.info('step: {}, {}'.format(self.state.global_step, self.acc_meter))

        #     self._reset_meters_if_needed()


    def _reset_meters_if_needed(self):
        if int(self.state.epoch) != self.last_epoch:#新的epoch则进入
            self.last_epoch = int(self.state.epoch)
            self.acc_meter.reset()
        if self.args.save_strategy == IntervalStrategy.STEPS and self.state.global_step % self.args.save_steps == 0:
            self.acc_meter.reset()
