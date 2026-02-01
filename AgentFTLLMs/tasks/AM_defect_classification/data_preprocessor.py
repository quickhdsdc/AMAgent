import pandas as pd
from typing import Dict, Any
from datasets import Dataset

LABEL_ORDER = ["none", "lof", "balling", "keyhole"]  # class ids 0,1,2,3

def _canon_label_text_from_int(x):
    try:
        xi = int(x)
    except Exception:
        return "none"
    if 0 <= xi < len(LABEL_ORDER):
        return LABEL_ORDER[xi]
    return "none"

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

    # ---------- message builders ----------

    def _build_system_msg(self) -> Dict[str, str]:
        """
        Global instruction/behavior for the assistant.
        This will be reused for every sample.
        """
        return {
            "role": "system",
            "content": (
                'You are a Laser Powder Bed Fusion (LPBF) process analysis assistant and act as an LPBF '
                'defect classification model. Given a set of process parameters, return ONLY the defect '
                'label that is one of "none", "lof", "balling", and "keyhole". '
                "Return ONLY the schema below. No extra text, no commentary, no explanations."
                "[LABEL] {one of \"none\", \"lof\", \"balling\", and \"keyhole\"} [/LABEL]"
                "[THINK] {concise justification} [/THINK]\n"
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
            "Your task is to assess in detail the potential imperfections for Laser Powder Bed Fusion printing "
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

    # ---------- row mappers ----------

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

        # Make sure numeric + canonical form are present
        sample["label"] = int(sample["label"])
        sample["label_text"] = _canon_label_text_from_int(sample["label"])

        messages = [
            self._build_system_msg(),
            self._build_user_msg(sample),
            self._build_assistant_msg_gt(sample),
        ]

        # serialize to llama-style chat text using its chat_template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # we include the assistant turn explicitly
        )

        sample["text"] = text

        # Optional: pre-tokenize for convenience
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
        enforce_structured_output: bool = False,
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
            # NOTE: we do NOT append assistant(gt) here
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # model should now generate the assistant reply
        )

        if enforce_structured_output:
            # Pre-fill the assistant response with [LABEL] to force structured output
            text += "[LABEL]"

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

    # ---------- dataset-level wrapper ----------

    def preprocess_data(
        self,
        tokenizer,
        dataset: Dataset,
        is_train: bool = True,
        max_length: int = 1024,
        enforce_structured_output: bool = False,
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
                return self._row_to_chat_text_eval(ex, tokenizer, max_length, enforce_structured_output)

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

        # We now only keep final columns we care about for training/eval
        # You could also filter here if you want.
        if is_train:
            processed = processed.shuffle(seed=42)

        return processed

