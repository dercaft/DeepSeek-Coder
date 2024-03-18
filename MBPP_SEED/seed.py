import os
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
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == '__main__':

    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=kwargs_handlers)   


    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument("--dataroot", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    args = parser.parse_args()

    logdir = args.logdir
    model_path = args.model_path
    if logdir == "":
        logdir = "tmp/"
    tokenizer = dict(
        cls=AutoTokenizer,
        model_path=model_path,)

    dataroot = args.dataroot

    evaluator = evaltor(data_root=dataroot, max_seq_len=4096, tokenizer_cfg=tokenizer, log_dir=logdir, n_sample=1, batch_size=1, max_gen_len=500)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=accelerator.device, trust_remote_code=True, ).bfloat16()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # evaluator.eval_model(model, accelerator)
    # 1. Error Code Collection
    status_with_err_codes=evaluator.eval_and_collect_error_code(model, accelerator)
    print("Status with Error Codes: ", len(status_with_err_codes))
    for i, (k,status_with_err_code) in enumerate(status_with_err_codes.items()):
        print(f"##Status with Error Code {i}: ", k,status_with_err_code)
        print(status_with_err_code["error_reason"])
    # Strcture of status_with_err_code:
    # {
    #     "prompt": "You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\nassert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)\nassert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)\n[BEGIN]\ndef similar_elements(test_tup1, test_tup2):\n  res = tuple(set(test_tup1) & set(test_tup2))\n  return (res) \n[DONE]\nYou are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:\n\nassert is_not_prime(2) == False\nassert is_not_prime(10) == True\nassert is_not_prime(35) == True\n[BEGIN]\nimport math\ndef is_not_prime(n):\n    result = False\n    for i in range(2,int(math.sqrt(n)) + 1):\n        if n % i == 0:\n            result = True\n    return result\n[DONE]\nYou are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:\n\nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]\n[BEGIN]\nimport heapq as hq\ndef heap_queue_largest(nums,n):\n  largest_nums = hq.nlargest(n, nums)\n  return largest_nums\n[DONE]\nYou are an expert Python programmer, and here is your task: Write a function to get a lucid number smaller than or equal to n. Your code should pass these tests:\n\nassert get_ludic(10) == [1, 2, 3, 5, 7]\nassert get_ludic(25) == [1, 2, 3, 5, 7, 11, 13, 17, 23, 25]\nassert get_ludic(45) == [1, 2, 3, 5, 7, 11, 13, 17, 23, 25, 29, 37, 41, 43]\n[BEGIN]",
    #     "answer": "def get_ludic(n):\r\n\tludics = []\r\n\tfor i in range(1, n + 1):\r\n\t\tludics.append(i)\r\n\tindex = 1\r\n\twhile(index != len(ludics)):\r\n\t\tfirst_ludic = ludics[index]\r\n\t\tremove_index = index + first_ludic\r\n\t\twhile(remove_index < len(ludics)):\r\n\t\t\tludics.remove(ludics[remove_index])\r\n\t\t\tremove_index = remove_index + first_ludic - 1\r\n\t\tindex += 1\r\n\treturn ludics",
    #     "generation": "def get_ludic(n):\n  ludic = []\n  for i in range(1,n+1):\n    ludic.append(i)\n  return (ludic)",
    #     "error_reason": ["failed: AssertionError "],
    # }

    
        # break
    # 2. Automatic Code Revision
    template='''
        Requirement: {}\n
        Correct Solution: {}\n
        Error Code: {}\n
        Failed Test Cases: {}\n
        Error Messages: {}\n
        Revised Code: {}\n
        '''
    # 3. Model Optimization