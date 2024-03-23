import os
import numpy as np
import json

class MBPPDataset:

    def __init__(self, root, samplenum=1,split="test"):
        """
        root: 数据文件的根目录
        """
        self.root = root
        self.split = split
        self.data = open(os.path.join(root, "mbpp.jsonl")).readlines()

        self.clean_data = self.get_qa_only_data(self.data)
        self.prompt = []
        for i in range(1, 4):            
            prompt = self.clean_data[i]["prompt"]
            tests = "\n".join(self.clean_data[i]["test"])
            code = self.clean_data[i]["code"].replace("\r", "").replace("\t", "    ")
            prompt1 = f"You are an expert Python programmer, and here is your task: {prompt} Your code should pass these tests:\n\n{tests}\n[BEGIN]\n{code}\n[DONE]\n"
            if len(self.prompt) == 0:
                self.prompt.append(prompt1)
            else:
                self.prompt.append(self.prompt[-1] + prompt1)
        self.testdata = []
        for i in range(10, 510):
            for j in range(samplenum):
                self.testdata.append(self.clean_data[i])
        self.traindata = []
        # for i in range(601, 957):
        for i in range(601, 957):
            self.traindata.append(self.clean_data[i])
        np.random.seed(1234)
        print(f"Read MBPP from {root}, number of samples {len(self.testdata)}")

    def get_qa_only_data(self, data_json):
        ans = []
        for line in data_json:
            line = json.loads(line)
            prompt = line["text"]
            suffix = line["test_list"]
            code = line["code"]
            ans.append({"prompt":prompt, "test":suffix, "code":code, "task_id":line["task_id"]})
        return ans

    def __len__(self):
        if self.split == "train":
            return len(self.traindata)
        elif self.split == "test":
            return len(self.testdata)

    def __getitem__(self, index):
        if self.split == "train":
            sample = self.traindata[index]
        elif self.split == "test":
            sample = self.testdata[index]
        return sample

