# model_loader_seqcls.py
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from accelerate import Accelerator
from dotenv import load_dotenv

load_dotenv()
HF_token = os.getenv("HFTOKEN")

LABEL_ORDER = ["none", "lof", "balling", "keyhole"]  # indices 0..3

class ModelLoader:
    def __init__(self, use_4bit: bool = True, accelerator: Accelerator = None) -> None:
        """
        Loader for SEQUENCE CLASSIFICATION over LPBF parameter strings.
        By default loads weights in 4-bit NF4 to save memory.
        """
        self.accelerator = accelerator
        self.bnb_config = None
        if use_4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

    def load_model_from_path_name_version(
        self,
        model_root_path: str,     # e.g. "meta-llama/Llama-3.1-8B"
        model_name: str,          # for logging
        model_version: str,       # for logging
        device_map: str = None, 
        num_labels: int = 4,
        label_order = LABEL_ORDER,
    ):
        """
        Returns: (model, tokenizer) ready for sequence classification training/eval.
        """
        
        if device_map is None:
            if self.accelerator:
                dev_idx = self.accelerator.process_index
                device_map = {"": dev_idx}
                # CRITICAL: bnb sometimes allocates on current device. Ensure it's set.
                if torch.cuda.is_available():
                    torch.cuda.set_device(dev_idx)
            else:
                device_map = "auto"

        if not self.accelerator or self.accelerator.is_main_process:
            print(
                f"Loading SEQ-CLS model:\n"
                f"  name={model_name}\n"
                f"  version={model_version}\n"
                f"  source={model_root_path}\n"
                f"  device_map={device_map}\n"
                f"  num_labels={num_labels}\n"
            )

        # 1) Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_root_path,
            token=HF_token,
        )
        # Ensure a pad token (decoder-only models often miss this)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # 2) Label maps
        id2label = {i: lab for i, lab in enumerate(label_order)}
        label2id = {lab: i for i, lab in enumerate(label_order)}

        # 3) Model (sequence classification head)
        model_kwargs = dict(
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            problem_type="single_label_classification",
        )

        if self.bnb_config is not None:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_root_path,
                token=HF_token,
                quantization_config=self.bnb_config,
                device_map=device_map,
                **model_kwargs,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_root_path,
                token=HF_token,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                **model_kwargs,
            )

        # Make sure model knows the pad token
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        # For decoder-only backbones, disabling cache helps training stability
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        
        if not self.accelerator or self.accelerator.is_main_process:
            print("Model and tokenizer loaded for SEQ-CLS.")
            print(
                f"- vocab size: {tokenizer.vocab_size}\n"
                f"- pad_token: {repr(tokenizer.pad_token)} (id={tokenizer.pad_token_id})\n"
                f"- eos_token: {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})\n"
                f"- id2label: {model.config.id2label}\n"
            )

        return model, tokenizer
