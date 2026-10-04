import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
from dotenv import load_dotenv
import os
load_dotenv()
HF_token = os.getenv("HFTOKEN")


class ModelLoader:
    def __init__(self, accelerator: Accelerator = None, load_in_4bit: bool = True) -> None:
        """
        Quantized model loader for causal LMs.
        - If load_in_4bit=True (default): Loads in 4-bit NF4 (Data Parallel friendly).
        - If load_in_4bit=False: Loads in bfloat16 (Model Parallel friendly).
        """
        self.accelerator = accelerator
        self.load_in_4bit = load_in_4bit
        
        if self.load_in_4bit:
            msg = "Initializing ModelLoader with 4-bit quant config..."
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            msg = "Initializing ModelLoader with bfloat16 config (NO quantization)..."
            self.bnb_config = None
            
        if not self.accelerator or self.accelerator.is_main_process:
             print(msg)

    def load_model_from_path_name_version(
        self,
        model_root_path: str,
        model_name: str,
        model_version: str,
        device_map: str = None, 
    ):
        """
        model_root_path: HF repo ID or local path, e.g. "meta-llama/Llama-3.1-8B"
        model_name/model_version: mostly for logging/bookkeeping
        device_map: "auto" or specific dict.
        """
        
        if device_map is None:
            if self.load_in_4bit:
                if self.accelerator:
                    device_map = {"": self.accelerator.process_index}
                else:
                    device_map = "auto"
            else:
                device_map = "auto"

        if not self.accelerator or self.accelerator.is_main_process:
            print(
                f"Loading model:\n"
                f"  name={model_name}\n"
                f"  version={model_version}\n"
                f"  source={model_root_path}\n"
                f"  device_map={device_map}\n"
                f"  load_in_4bit={self.load_in_4bit}\n"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_root_path,
            token=HF_token,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            model_root_path,
            token=HF_token,
            quantization_config=self.bnb_config,
            device_map=device_map,
            torch_dtype=None if self.load_in_4bit else torch.bfloat16,
        )

        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id


        if not self.accelerator or self.accelerator.is_main_process:
            print("Model and tokenizer loaded.")
            print(
                f"- vocab size: {tokenizer.vocab_size}\n"
                f"- pad_token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})\n"
                f"- eos_token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})\n"
            )

        return model, tokenizer