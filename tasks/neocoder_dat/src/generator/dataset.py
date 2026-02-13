"""Prepare data for inference
"""
from torch.utils.data import Dataset
import json

class DPInferenceDataset(Dataset):
    def __init__(self, 
                 data_path,
                 dp_rounds):
        super().__init__()
        self.data = self.load_problem_json(data_path, dp_rounds)

    def load_problem_json(self, path, dp_rounds):
        with open(path, "r") as f:
            data = json.load(f)

        # # filter out problems that have less than dp_rounds constrains
        # data = [d for d in data if max(list(map(len, d['constraints_list']))) >= dp_rounds]

        for item in data:
            # include the og problem
            item['problem_statements'] = item['problem_statements'][:dp_rounds+1]
            item['constraints_list'] = item['constraints_list'][:dp_rounds+1]
            item['codes'] = item['codes'][:dp_rounds+1]
            
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        return self.data[index]
