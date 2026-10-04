from tasks.AM_defect_classification.data_preprocessor import DataPreprocessor
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer

class MockTokenizer:
    def __init__(self):
        self.chat_template = "mock_template"
        self.pad_token_id = 0
        self.eos_token_id = 1
    
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        res = ""
        for m in messages:
            res += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            res += "<|im_start|>assistant\n"
        return res

    def __call__(self, text, **kwargs):
        return {"input_ids": [0, 1, 2], "attention_mask": [1, 1, 1]}

def verify_prompts():
    prep = DataPreprocessor()
    tokenizer = MockTokenizer()
    
    data = [{
        "label": 0,
        "material": "SS17-4PH",
        "Power": 1000.0,
        "Velocity": 8000.0,
        "beam D": 50.0,
        "layer thickness": 40.0
    }]
    ds = Dataset.from_pandas(pd.DataFrame(data))
    
    print("\n--- TRAIN PROMPT ---")
    train_ds = prep.preprocess_data(tokenizer, ds, is_train=True, max_length=128)
    print(train_ds[0]["text"])
    
    print("\n--- EVAL PROMPT ---")
    eval_ds = prep.preprocess_data(tokenizer, ds, is_train=False, max_length=128)
    print(eval_ds[0]["text"])

if __name__ == "__main__":
    verify_prompts()
