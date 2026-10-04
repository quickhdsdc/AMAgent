import pandas as pd
from typing import Dict, Any
from datasets import Dataset

from .task_labels import LABEL_ORDER


def _canon_label_text_from_int(x):
    try:
        xi = int(x)
    except Exception:
        return LABEL_ORDER[0]
    if 0 <= xi < len(LABEL_ORDER):
        return LABEL_ORDER[xi]
    return LABEL_ORDER[0]


def _label_choices_str() -> str:
    """Human-readable quoted list of the active label set, e.g.
    '"none", "lof", "balling", and "keyhole"'  or  '"good" and "defective"'.
    """
    quoted = [f'"{lbl}"' for lbl in LABEL_ORDER]
    if len(quoted) == 1:
        return quoted[0]
    if len(quoted) == 2:
        return f"{quoted[0]} and {quoted[1]}"
    return ", ".join(quoted[:-1]) + ", and " + quoted[-1]

def _safe_val(x):
    if pd.isna(x):
        return "unknown"
    return str(x)


class DataPreprocessor:
    """
    AM LPBF defect classification prompt builder using the model's chat_template.

    We produce conversational message dicts:
      - system: global instruction ("you are LPBF assistant...")
      - user:   per-row LPBF parameters and classification question
      - assistant: gold label (only in training mode)

    Then we serialize with tokenizer.apply_chat_template(...).

    Workflow:
    - preprocess_data(..., is_train=True):
        -> build [system,user,assistant(gt)]
        -> add_generation_prompt=False
        -> model sees gold answer in text
    - preprocess_data(..., is_train=False):
        -> build [system,user]
        -> add_generation_prompt=True
        -> model will generate assistant turn at eval time

    Each row after mapping will contain:
        sample["text"]              # serialized conversation string
        sample["label"]             # int class id (0..3)
        sample["label_text"]        # canonical class string
        sample["input_ids_text"]    # token ids tensor (optional)
        sample["attention_mask_text"]
    """

    def __init__(self) -> None:
        print("Preprocessing the data...")


    def _build_system_msg(self) -> Dict[str, str]:
        """
        Global instruction/behavior for the assistant.
        This will be reused for every sample.
        """
        choices = _label_choices_str()
        return {
            "role": "system",
            "content": (
                'You are a Laser Powder Bed Fusion (LPBF) process analysis assistant and act as an LPBF '
                'defect classification model. Given a set of process parameters, return exactly one '
                f'label from {choices}. Output only the label word, with no tags or explanation.'
            ),
        }

    def _build_user_msg(self, row: Dict[str, Any]) -> Dict[str, str]:
        """
        User prompt describing this specific LPBF condition.
        We will fill in the measured parameters + ask for the label.
        """
        mat = row.get("material", "unknown material")
        pwr = _safe_val(row.get("Power"))
        vel = _safe_val(row.get("Velocity"))
        bd  = _safe_val(row.get("beam D"))
        lt  = _safe_val(row.get("layer thickness"))

        user_content = (
            "Classify the most likely defect outcome for Laser Powder Bed Fusion printing "
            f"that arise in {mat} manufactured at {pwr} W, utilizing a {bd} µm beam, "
            f"traveling at {vel} mm/s, with a layer thickness of {lt} µm. "
            "Predict the potential defect label."
        )

        return {
            "role": "user",
            "content": user_content,
        }

    def _build_assistant_msg_gt(self, row: Dict[str, Any]) -> Dict[str, str]:
        """
        Assistant message that contains ONLY the gold label text.
        No reasoning, no brackets, just e.g. 'keyhole'
        """
        label_text = row.get("label_text")
        if label_text is None:
            label_text = _canon_label_text_from_int(row.get("label", 0))

        return {
            "role": "assistant",
            "content": label_text,
        }


    def _row_to_chat_text_train(
        self,
        sample: Dict[str, Any],
        tokenizer,
        max_length: int,
    ) -> Dict[str, Any]:
        """
        TRAIN FORMAT:
        messages = [system, user, assistant(gt)]
        add_generation_prompt = False
        => the serialized text ends with the gold label turn.
        """

        sample["label"] = int(sample["label"])
        sample["label_text"] = _canon_label_text_from_int(sample["label"])

        messages = [
            self._build_system_msg(),
            self._build_user_msg(sample),
            self._build_assistant_msg_gt(sample),
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        sample["text"] = text

        enc = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        sample["input_ids_text"] = enc["input_ids"]
        sample["attention_mask_text"] = enc["attention_mask"]

        return sample

    def _row_to_chat_text_eval(
        self,
        sample: Dict[str, Any],
        tokenizer,
        max_length: int,
    ) -> Dict[str, Any]:
        """
        EVAL / INFERENCE FORMAT:
        messages = [system, user]
        add_generation_prompt = True
        => the serialized text ends right before the assistant reply header,
           so generation should produce ONLY the label.

        We still keep ground truth label_text in the row for scoring.
        """

        sample["label"] = int(sample["label"])
        sample["label_text"] = _canon_label_text_from_int(sample["label"])

        messages = [
            self._build_system_msg(),
            self._build_user_msg(sample),
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        sample["text"] = text

        enc = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        sample["input_ids_text"] = enc["input_ids"]
        sample["attention_mask_text"] = enc["attention_mask"]

        return sample


    def preprocess_data(
        self,
        tokenizer,
        dataset: Dataset,
        is_train: bool = True,
        max_length: int = 1024,
    ) -> Dataset:
        """
        Map over a HuggingFace Dataset split and build chat-formatted 'text'
        using tokenizer.apply_chat_template().
        - If is_train=True: include assistant's gold label turn
        - If is_train=False: only system+user and add_generation_prompt=True

        Returns a new Dataset with:
          text, label, label_text, input_ids_text, attention_mask_text
        """

        print("Preprocessing dataset... (is_train =", is_train, ")")
        self.max_length = max_length

        if is_train:
            def map_fn(ex):
                return self._row_to_chat_text_train(ex, tokenizer, max_length)
        else:
            def map_fn(ex):
                return self._row_to_chat_text_eval(ex, tokenizer, max_length)

        processed = dataset.map(
            map_fn,
            remove_columns=[
                col for col in dataset.column_names
                if col not in [
                    "label",
                    "material",
                    "Power",
                    "Velocity",
                    "beam D",
                    "layer thickness",
                ]
            ],
            keep_in_memory=True,
        )

        if is_train:
            processed = processed.shuffle(seed=42)

        return processed

