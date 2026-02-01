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
                load_in_4bit=True,                # 4-bit base weights
                bnb_4bit_use_double_quant=True,   # nested quantization
                bnb_4bit_quant_type="nf4",        # nf4 quantization
                bnb_4bit_compute_dtype=torch.bfloat16,  # matmul compute dtype
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
        
        # Determine device_map
        if device_map is None:
            if self.load_in_4bit:
                 # Case 1: 4-bit loading (DDP) -> Map to specific process device
                if self.accelerator:
                    device_map = {"": self.accelerator.process_index}
                else:
                    device_map = "auto"
            else:
                # Case 2: bfloat16 loading (MP) -> Use "auto" to shard across GPUs
                # This REQUIRES running with num_processes=1 if the model > single GPU VRAM.
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

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_root_path,
            token=HF_token,
        )

        # Most decoder-only LLMs (Llama, etc.) don't define a pad token
        # We need one for batching/Trainer, so we reuse eos_token.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_root_path,
            token=HF_token,
            quantization_config=self.bnb_config,
            device_map=device_map,
            torch_dtype=None if self.load_in_4bit else torch.bfloat16,
        )

        # Make sure pad_token_id is consistent in the model config too
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        # Optional quality-of-life: trust remote code (if needed for some models),
        # but for llama3.1 we usually don't need trust_remote_code=True.

        if not self.accelerator or self.accelerator.is_main_process:
            print("Model and tokenizer loaded.")
            print(
                f"- vocab size: {tokenizer.vocab_size}\n"
                f"- pad_token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})\n"
                f"- eos_token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})\n"
            )

        return model, tokenizer