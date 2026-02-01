import os
from pathlib import Path
from datasets import load_dataset
from typing import Callable, Dict, List, Any


class TaskDataLoader:
    def __init__(self, task_name:str, train_type:str, val_type:str, test_type:str) -> None:
        '''
        params:task_name: str: name of the task
        params:train_type: str: type of the training data in ['train','train_small']
        params:val_type: str: type of the validation data in ['test']
        params:test_type: str: type of the test data in ['test','test_big']
        '''
        self.task_name = task_name
        self.train_type = train_type
        self.test_type = test_type
        self.task_data_dir = Path('data/Re3-Sci/tasks') / task_name
        data_files = {}
        for i in data_files.iterdir():
            if i.is_file() and i.name.endswith('.csv'):
                data_files[i.stem] = str(i)
        self.data_files = data_files
        self.dataset = load_dataset("csv", data_files = data_files)
        print('dataset', self.dataset)
        self.labels = self.dataset['train'].features['label'].names
        print('labels', self.labels)
