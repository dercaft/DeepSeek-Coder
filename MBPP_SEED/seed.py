import os
from MBPP_SEED.seed_datasets import CodeDataset, load_seed_datasets,ReviseDataset
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json
import torch.distributed as dist
import subprocess
import sys
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from pathlib import Path
from argparse import ArgumentParser
from mbpp import MBPP as evaltor
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from typing import Dict, Optional, Sequence
from transformers.trainer_pt_utils import LabelSmoother
from transformers import Trainer
from dataclasses import dataclass, field
from torch.utils.data import ConcatDataset
from peft import LoraConfig, TaskType, get_peft_model

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_TOKEN_ID
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(
    logdir,timestamp,tokenizer,dp_size,use_revised_data=False
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    l_datasets = []
    if use_revised_data:
        l_datasets.append(ReviseDataset(logdir,timestamp,tokenizer,dp_size))
    else:
        l_datasets.append(CodeDataset(logdir,timestamp,tokenizer,dp_size))
    train_dataset = ConcatDataset(datasets=l_datasets)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

if __name__ == '__main__':
    timestamp =datetime.now().strftime("%Y%m%d_%H")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=kwargs_handlers)   
    dp_rank = accelerator.process_index
    dp_size = accelerator.num_processes
    print(dp_rank)

    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument("--dataroot", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--model_max_length", type=int, default=6000)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    args = parser.parse_args()

    logdir = args.logdir
    model_path = args.model_path
    if logdir == "":
        logdir = "tmp/"
    
    dataroot = args.dataroot
    max_seq_len=args.model_max_length
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=accelerator.device, trust_remote_code=True, ).bfloat16()
    revise_model = AutoModelForCausalLM.from_pretrained(model_path, device_map=accelerator.device, trust_remote_code=True, ).bfloat16().cpu()
    for iteration in range(3):
        tokenizer = dict(cls=AutoTokenizer,model_path=model_path,)
        timestamp_iteration=f"{timestamp}_iter{iteration}"
        loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer["model_path"])
        evaluator = evaltor(data_root=dataroot, max_seq_len=4096,\
                            tokenizer_cfg=tokenizer, log_dir=logdir, n_sample=1,\
                                batch_size=1, max_gen_len=300,temperature=0.5)
        
        model.to(dp_rank)
        log_file=evaluator.eval_and_collect_error_code(model, accelerator)
        model.cpu()

        revise_model.to(dp_rank)
        results=evaluator.icl_revise_error_code(revise_model, accelerator, log_file, timestamp_iteration)
        accelerator.wait_for_everyone()
        revise_model.cpu()
        # 3. Model Optimization
        # 3.1 optimize the revise_model
        
        training_args=transformers.TrainingArguments(output_dir=logdir,per_device_train_batch_size =1,gradient_checkpointing=True)
        data_module = make_supervised_data_module(logdir,timestamp_iteration,loaded_tokenizer,dp_size,use_revised_data=True)
        # trainer = Trainer(
        #     model=revise_model, tokenizer=loaded_tokenizer, args=training_args, **data_module
        # )
        # trainer.train()
        # revise_model.cpu()

        # 3.2 optimize the code_model 
        data_module = make_supervised_data_module(logdir,timestamp_iteration,loaded_tokenizer,dp_size,use_revised_data=False)
        trainer = Trainer(
            model=model, tokenizer=loaded_tokenizer, args=training_args, **data_module
        )
        trainer.train()
        model.cpu()

        if dp_rank == 0:
            subdir = f"saved_model_iter{iteration}"
            full_path = os.path.join(args.logdir, subdir)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            model.save_pretrained(full_path)
            loaded_tokenizer.save_pretrained(full_path)
            subdir = f"saved_revise_model_iter{iteration}"
            full_path = os.path.join(args.logdir, subdir)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            revise_model.save_pretrained(full_path)
            loaded_tokenizer.save_pretrained(full_path)
            


    

