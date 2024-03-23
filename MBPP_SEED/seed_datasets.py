import copy
from datasets import Dataset
import json
import os
from dataclasses import dataclass, field
from transformers.trainer_pt_utils import LabelSmoother
import torch

def load_json_dataset(file_path):
    """Load data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
        dataset = Dataset.from_dict({k: [dic[k] for dic in data] for k in data[0]})
        return dataset

def last_index(lst, value):
    return next((len(lst) - i - 1 for i, x in enumerate(lst[::-1]) if x != value), -1)


def safe_ids(ids, max_value, pad_id):
    return [i if i < max_value else pad_id for i in ids]


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# B_INST, E_INST = "[INST]", "[/INST]"
# B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
B_INST, E_INST="",""
B_SYS, E_SYS="",""
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def _tokenize(item, tokenizer):
    roles = {"human": "user", "gpt": "assistant"}
    input_ids = []
    labels = []
    # if "instruction" in item and len(item["instruction"]) > 0:
    #     system = item["instruction"]
    # else:
    system = item["system"]
    # raise ValueError("instruction is empty")
    system = B_SYS + system + E_SYS
    # add system before the first content in conversations
    item["conversations"][0]["value"] = (
        system + "\n\n" + item["conversations"][0]["value"]
    )
    # item["input"] = system + item["input"]
    for i, turn in enumerate(item["conversations"]):
        role = turn["from"]
        content = turn["value"]
        content = content.strip()
        if role == "human":
            content = f"{B_INST}{content}{E_INST} "
            content_ids = tokenizer.encode(content)
            labels += [IGNORE_TOKEN_ID] * (len(content_ids))
        else:
            # assert role == "gpt"
            content = f"{content}"
            content_ids = tokenizer.encode(content, add_special_tokens=False) + [
                tokenizer.eos_token_id
            ]  # add_special_tokens=False remove bos token, and add eos at the end
            labels += content_ids
        input_ids += content_ids

    input_ids = input_ids[: tokenizer.model_max_length]
    labels = labels[: tokenizer.model_max_length]

    trunc_id = last_index(labels, IGNORE_TOKEN_ID) + 1
    input_ids = input_ids[:trunc_id]
    labels = labels[:trunc_id]
    if len(labels) == 0:
        # return RAFTDataset._tokenize(RAFTDataset.dummy_message, tokenizer)
        assert False, "labels is empty"
    input_ids = safe_ids(input_ids, tokenizer.vocab_size, tokenizer.pad_token_id)
    labels = safe_ids(labels, tokenizer.vocab_size, IGNORE_TOKEN_ID)
    return input_ids, labels


class CodeDataset(Dataset):
    def __init__(self,logdir,timestamp,tokenizer,dp_size ):
        self.dataset = []
        for dp_rank in range(dp_size):
            filename = f"revised_{timestamp}_rank{dp_rank}.jsonl"
            filename = os.path.join(logdir,filename)
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as jsonlfile:
                    for line in jsonlfile:
                        result = json.loads(line)
                        self.dataset.append(result)
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        messages=[{"from": "human", "value":self.dataset[index]['code_prompt'][0]+"[DONE]"},
                {"from": "gpt", "value": self.dataset[index]['revised_code'][0]+"[DONE]"}]
        system=""
        chosen_messages=copy.deepcopy(messages)
        chosen_item = {"conversations": chosen_messages, "system": system}
        chosen_input_ids, chosen_labels = _tokenize(copy.deepcopy(chosen_item), self.tokenizer)
        input_ids = torch.tensor(chosen_input_ids)
        labels = torch.tensor(chosen_labels)
        return dict(
            input_ids=input_ids,
            labels=labels,
        )

class ReviseDataset(CodeDataset):
    def __getitem__(self, index):
        messages=[{"from": "human", "value":self.dataset[index]['revise_prompt']+"[DONE]"},
                {"from": "gpt", "value": self.dataset[index]['revised_code']+"[DONE]"}]
        system=""
        chosen_messages=copy.deepcopy(messages)
        chosen_item = {"conversations": chosen_messages, "system": system}
        chosen_input_ids, chosen_labels = _tokenize(copy.deepcopy(chosen_item), self.tokenizer)
        input_ids = torch.tensor(chosen_input_ids)
        labels = torch.tensor(chosen_labels)
        return dict(
            input_ids=input_ids,
            labels=labels,
        )

def load_seed_datasets(tokenizer, max_length, logdir, accelerator, timestamp):
    dp_size = accelerator.num_processes
    results = []
    for dp_rank in range(dp_size):
        filename = f"revised_{timestamp}_rank{dp_rank}.jsonl"
        filename = os.path.join(logdir,filename)
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as jsonlfile:
                for line in jsonlfile:
                    result = json.loads(line)
                    results.append(result)
    code_dataset = Dataset.from_dict({k: [dic[k] for dic in results] for k in results[0]})
    revise_dataset = Dataset.from_dict({k: [dic[k] for dic in results] for k in results[0]})
    def code_preprocess_function(result):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        messages=[{"from": "human", "value":result['code_prompt'][0]+"[DONE]"},
                {"from": "gpt", "value": result['revised_code'][0]+"[DONE]"}]
        system=""
        chosen_messages=copy.deepcopy(messages)
        chosen_item = {"conversations": chosen_messages, "system": system}
        chosen_input_ids, chosen_labels = _tokenize(copy.deepcopy(chosen_item), tokenizer)
            # new_examples["input_ids_chosen"].append(chosen_input_ids)
            # new_examples["attention_mask_chosen"].append([1]*len(chosen_labels))
        return dict(
            input_ids=chosen_input_ids,
            labels=chosen_labels,
        )
    
    def revise_preprocess_function(result):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        # for result in results:
        messages=[{"from": "human", "value":result['revise_prompt'][0]+"[DONE]"},
                    {"from": "gpt", "value": result['revised_code'][0]+"[DONE]"}]
        system=""
        chosen_messages=copy.deepcopy(messages)
        chosen_item = {"conversations": chosen_messages, "system": system}
        chosen_input_ids, chosen_labels = _tokenize(copy.deepcopy(chosen_item), tokenizer)
            # new_examples["input_ids_chosen"].append(chosen_input_ids)
            # new_examples["attention_mask_chosen"].append([1]*len(chosen_labels))
        return dict(
            input_ids=chosen_input_ids,
            labels=chosen_labels,
        )
        
    code_dataset = code_dataset.map(code_preprocess_function,batched=True,num_proc=4,)
    # code_dataset = code_dataset.filter(lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length)
    revise_dataset = revise_dataset.map(revise_preprocess_function,batched=True,num_proc=4,)
    # revise_dataset = revise_dataset.filter(lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length)
    return code_dataset,revise_dataset




example1="""As a code correction expert, you will be given incorrect code and the reasons for the errors.  Your objective is to make the minimal necessary modifications to the provided code based on the task requirements and detailed reasons for the errors.  \n\n            Question:You are an expert Python programmer, and here is your task: Write a python function to find the minimum number of rotations required to get the same string. Your code should pass these tests:\n\nassert find_Rotations("aaaa") == 1\nassert find_Rotations("ab") == 2\nassert find_Rotations("abc") == 3\n\n\n            Correct Solution: def find_Rotations(str): \r\n    tmp = str + str\r\n    n = len(str) \r\n    for i in range(1,n + 1): \r\n        substring = tmp[i: i+n] \r\n        if (str == substring): \r\n            return i \r\n    return n \n\n            Error Code: import itertools\ndef find_Rotations(string):\n    result = 1\n    for i in itertools.product(range(len(string)), repeat=result):\n        for j in itertools.product(range(len(string)), repeat=result):\n            result -= 1\n    return result\n\n            Error Messages: [\'failed: \\n  File "<string>", line 8, in <module>\\n  File "<string>", line 5, in find_Rotations\\nValueError: repeat argument cannot be negative\\n\']\n\n            Revised Code: [BEGIN]\ndef find_Rotations(str): \n    tmp = str + str\n    n = len(str) \n    for i in range(1, n + 1): \n        substring = tmp[i: i+n] \n        if (str == substring): \n            return i \n    return n\n[DONE]"""
example2="""As a code correction expert, you will be given incorrect code and the reasons for the errors.  Your objective is to make the minimal necessary modifications to the provided code based on the task requirements and detailed reasons for the errors.  \n\n            Question:You are an expert Python programmer, and here is your task: Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][]. Your code should pass these tests:\n\nassert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8\nassert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12\nassert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16\n\n\n            Correct Solution: R = 3\r\nC = 3\r\ndef min_cost(cost, m, n): \r\n\ttc = [[0 for x in range(C)] for x in range(R)] \r\n\ttc[0][0] = cost[0][0] \r\n\tfor i in range(1, m+1): \r\n\t\ttc[i][0] = tc[i-1][0] + cost[i][0] \r\n\tfor j in range(1, n+1): \r\n\t\ttc[0][j] = tc[0][j-1] + cost[0][j] \r\n\tfor i in range(1, m+1): \r\n\t\tfor j in range(1, n+1): \r\n\t\t\ttc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j] \r\n\treturn tc[m][n]\n\n            Error Code: def min_cost(cost_matrix, row_pos, column_pos):\n  result = 0\n  queue = []\n  for i in range(0,len(cost_matrix)):\n    item = cost_matrix[i][row_pos]\n    if item < cost_matrix[i][column_pos]:\n      queue.append(item)\n    else:\n      queue.append(cost_matrix[i][column_pos])\n  for item in queue:\n    result += item\n  return result\n\n            Error Messages: [\'failed: \\n  File "<string>", line 14, in <module>\\nAssertionError\\n\']\n\n            Revised Code: [BEGIN]\ndef min_cost(cost, m, n): \n    R = 3\n    C = 3\n    tc = [[0 for x in range(C)] for x in range(R)] \n    tc[0][0] = cost[0][0] \n    for i in range(1, m+1): \n        tc[i][0] = tc[i-1][0] + cost[i][0] \n    for j in range(1, n+1): \n        tc[0][j] = tc[0][j-1] + cost[0][j] \n    for i in range(1, m+1): \n        for j in range(1, n+1): \n            tc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j] \n    return tc[m][n]\n[DONE]"""
example3="""As a code correction expert, you will be given incorrect code and the reasons for the errors.  Your objective is to make the minimal necessary modifications to the provided code based on the task requirements and detailed reasons for the errors.  \n\n            Question:You are an expert Python programmer, and here is your task: Write a function to get the n smallest items from a dataset. Your code should pass these tests:\n\nassert small_nnum([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100],2)==[10,20]\nassert small_nnum([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100],5)==[10,20,20,40,50]\nassert small_nnum([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100],3)==[10,20,20]\n\n\n            Correct Solution: import heapq\r\ndef small_nnum(list1,n):\r\n  smallest=heapq.nsmallest(n,list1)\r\n  return smallest\n\n            Error Code: import heapq as hq\ndef small_nnum(data,n):\n  largest_nums = hq.nlargest(n, data)\n  return largest_nums\n\n            Error Messages: [\'failed: \\n  File "<string>", line 5, in <module>\\nAssertionError\\n\']\n\n            Revised Code: [BEGIN]\nimport heapq\n\ndef small_nnum(list1, n):\n    smallest = heapq.nsmallest(n, list1)\n    return smallest\n[DONE]"""