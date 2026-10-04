from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["AM_TASK"] = "binary"

import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


STEMS = ["Exp_OOD_5", "Exp_OOD_6", "Exp_OOD_7"]
FIELDNAMES = [
    "row_idx", "material", "Power", "Velocity", "beam D",
    "layer thickness", "Hatch spacing",
    "zs_response", "zs_label", "gt_label",
]
_LABEL_RE = re.compile(
    r"\[LABEL\]\s*([^\[\]]+?)\s*\[/LABEL\]",
    re.IGNORECASE | re.DOTALL,
)


def norm_binary(y) -> str:
    s = str(y).strip().lower()
    if s in ("good", "0", "0.0"):
        return "good"
    if s in ("defective", "1", "1.0", "bad", "defect"):
        return "defective"
    return "unknown"


def extract_binary_label(text: str) -> str:
    """Pull `good`/`defective` out of a `[LABEL] ... [/LABEL]` block."""
    if not isinstance(text, str):
        return "unknown"
    m = _LABEL_RE.search(text)
    if not m:
        return "unknown"
    raw = re.sub(r"[\{\}\"']", "", m.group(1)).strip().lower()
    if ":" in raw:
        raw = raw.split(":")[-1].strip()
    return norm_binary(raw)


def build_zs_prompt_binary(row) -> str:
    """Verbatim from `run_C8C9_llm_zs_binary.build_zs_prompt_binary`."""
    return (
        "You are an LPBF process analysis assistant. Assess whether a part "
        f"manufactured in {row['material']} by Laser Powder Bed Fusion at "
        f"{row['Power']} W, with a {row['beam D']} um beam, scan velocity "
        f"{row['Velocity']} mm/s, layer thickness {row['layer thickness']} um "
        f"and hatch spacing {row['Hatch spacing']} um is likely good or "
        "defective. A part is `good` if its relative density is at least "
        "99 % and no defect mechanism (lack of fusion, keyhole, balling) "
        "dominates; otherwise it is `defective`. Consider whether these "
        f"parameters respect the typical process window for {row['material']}, "
        "using only your internal physics knowledge.\n\n"
        "Return ONLY the schema below:\n"
        "[THINK] {reasoning} [/THINK]\n"
        "[LABEL] {one of \"good\", \"defective\"} [/LABEL]"
    )


PROFILE = "qwen3_local"
MODEL_KEY = "Qwen3-30B-A3B-Thinking-2507"
MODEL_PATH = os.environ.get("AM_QWEN3_THINKING_PATH", "Qwen/Qwen3-30B-A3B-Thinking-2507")

DATA_DIR = Path("data/_am_qc/data_exp")
OUT_ROOT = Path("results_AM") / f"AMagent_ZS_{PROFILE}"

BATCH_SIZE = 2
MAX_NEW_TOKENS = 8192


def out_path(stem: str) -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    return OUT_ROOT / f"ZS_{PROFILE}_preds_{stem}.csv"


def _load_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    try:
        return set(pd.read_csv(path)["row_idx"].astype(int))
    except Exception:
        return set()


def _append_rows(path: Path, batch: list[dict]) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        for r in batch:
            w.writerow(r)


def main() -> None:
    print(f"Loading {MODEL_KEY} from {MODEL_PATH} ...")
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        device_map={"": 0},
        max_memory={0: "75GiB"},
        low_cpu_mem_usage=True,
    )
    model.eval()
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    print(f"Model loaded in {time.time() - t0:.1f}s. Device: {model.device}")

    eos_ids = (
        getattr(model.generation_config, "eos_token_id", None)
        or tokenizer.eos_token_id
    )

    summary_rows = []
    for stem in STEMS:
        df = pd.read_csv(DATA_DIR / f"{stem}_test.csv")
        n = len(df)
        path = out_path(stem)
        done = _load_done(path)

        print(f"\n--- {stem}: {n} rows ({len(done)} done) profile={PROFILE} ---")

        todo = [(int(idx), row) for idx, row in df.iterrows() if int(idx) not in done]
        if not todo:
            print(f"  all rows already done; skipping inference for {stem}")
        else:
            user_prompts = [build_zs_prompt_binary(r) for _, r in todo]
            chat_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in user_prompts
            ]

            t_stem = time.time()
            for start in range(0, len(todo), BATCH_SIZE):
                batch_idx_rows = todo[start:start + BATCH_SIZE]
                batch_prompts = chat_prompts[start:start + BATCH_SIZE]

                enc = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to(model.device)

                with torch.no_grad():
                    outs = model.generate(
                        **enc,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        top_k=20,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=eos_ids,
                    )
                new_tokens = outs[:, enc.input_ids.shape[1]:]
                gens = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

                batch_csv_rows = []
                for (idx, row), gen in zip(batch_idx_rows, gens):
                    gt = norm_binary(row["defect_label"])
                    lbl = extract_binary_label(gen)
                    batch_csv_rows.append({
                        "row_idx": int(idx),
                        "material": row["material"],
                        "Power": row["Power"],
                        "Velocity": row["Velocity"],
                        "beam D": row["beam D"],
                        "layer thickness": row["layer thickness"],
                        "Hatch spacing": row["Hatch spacing"],
                        "zs_response": str(gen).replace("\n", " "),
                        "zs_label": lbl,
                        "gt_label": gt,
                    })
                _append_rows(path, batch_csv_rows)

                done_now = start + len(batch_idx_rows)
                if done_now % (BATCH_SIZE * 10) == 0 or done_now == len(todo):
                    rate = done_now / (time.time() - t_stem + 1e-6)
                    eta = (len(todo) - done_now) / rate if rate > 0 else float("inf")
                    print(f"  {done_now}/{len(todo)}  {rate:.2f} samp/s  ETA {eta/60:.1f} min")

        d = pd.read_csv(path)
        v = d[d.zs_label.isin(["good", "defective"]) & d.gt_label.isin(["good", "defective"])]
        f1 = (
            float(f1_score(v.gt_label, v.zs_label, average="macro"))
            if len(v) else float("nan")
        )
        acc = (
            float(accuracy_score(v.gt_label, v.zs_label))
            if len(v) else float("nan")
        )
        coverage = len(v) / len(d) if len(d) else float("nan")
        print(f"  {stem}: macro-F1 = {f1:.4f}  acc = {acc:.4f}  "
              f"coverage = {coverage:.3f}  (n_valid = {len(v)} / {len(d)})")

        summary_rows.append({
            "experiment": stem,
            "model_key": MODEL_KEY,
            "profile": PROFILE,
            "output_csv": str(path),
            "n_total": len(d),
            "n_valid": len(v),
            "coverage": f"{coverage:.4f}",
            "macro_f1": f"{f1:.4f}",
            "accuracy": f"{acc:.4f}",
        })

    pd.DataFrame(summary_rows).to_csv(
        OUT_ROOT / f"ZS_{PROFILE}_summary.csv", index=False,
    )
    print("\n=== FINAL SUMMARY (C8 zero-shot) ===")
    for r in summary_rows:
        print(f"  {r['experiment']}: macro-F1={r['macro_f1']}  acc={r['accuracy']}  "
              f"coverage={r['coverage']}")


if __name__ == "__main__":
    main()
