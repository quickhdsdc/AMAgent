import argparse
import csv
import os
import re
import sys

import pandas as pd
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from amcore import call_llm

STEMS = ["Exp_OOD_5", "Exp_OOD_6", "Exp_OOD_7"]
FIELDNAMES = ["row_idx", "material", "Power", "Velocity", "beam D",
              "layer thickness", "Hatch spacing",
              "zs_response", "zs_label", "gt_label"]
_LABEL_RE = re.compile(r"\[LABEL\]\s*([^\[\]]+?)\s*\[/LABEL\]",
                       re.IGNORECASE | re.DOTALL)


def norm_binary(y):
    s = str(y).strip().lower()
    if s in ("good", "0", "0.0"):
        return "good"
    if s in ("defective", "1", "1.0", "bad", "defect"):
        return "defective"
    return "unknown"


def extract_binary_label(text):
    if not isinstance(text, str):
        return "unknown"
    m = _LABEL_RE.search(text)
    if not m:
        return "unknown"
    raw = re.sub(r"[\{\}\"']", "", m.group(1)).strip().lower()
    if ":" in raw:
        raw = raw.split(":")[-1].strip()
    return norm_binary(raw)


def build_zs_prompt_binary(row):
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


def out_path(profile, stem):
    d = os.path.join(ROOT, "results_AM", f"AMagent_ZS_{profile}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"ZS_{profile}_preds_{stem}.csv")


def run_stem(stem, profile, limit=None):
    df = pd.read_csv(os.path.join(ROOT, "data_exp", f"{stem}_test.csv"))
    path = out_path(profile, stem)
    done = set()
    if os.path.exists(path):
        try:
            done = set(pd.read_csv(path)["row_idx"].astype(int))
        except Exception:
            pass
    print(f"\n--- {stem}: {len(df)} rows ({len(done)} done) profile={profile} ---")
    n_new = 0
    for idx, row in df.iterrows():
        if int(idx) in done:
            continue
        if limit is not None and n_new >= limit:
            break
        gt = norm_binary(row["defect_label"])
        try:
            resp = call_llm(build_zs_prompt_binary(row), profile=profile)
            lbl = extract_binary_label(resp)
        except Exception as e:
            resp, lbl = f"[ERROR] {e}", "unknown"
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                w.writeheader()
            w.writerow({"row_idx": int(idx), "material": row["material"],
                        "Power": row["Power"], "Velocity": row["Velocity"],
                        "beam D": row["beam D"],
                        "layer thickness": row["layer thickness"],
                        "Hatch spacing": row["Hatch spacing"],
                        "zs_response": str(resp).replace("\n", " "),
                        "zs_label": lbl, "gt_label": gt})
        n_new += 1
        print(f"  [{idx + 1}/{len(df)}] {row['material']} GT={gt} ZS={lbl}")
    d = pd.read_csv(path)
    v = d[d.zs_label.isin(["good", "defective"])
          & d.gt_label.isin(["good", "defective"])]
    f1 = f1_score(v.gt_label, v.zs_label, average="macro") if len(v) else float("nan")
    print(f"  {stem}: ZS macro-F1 = {f1:.3f}  (n_valid={len(v)})")
    return f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="gpt5")
    p.add_argument("--stem", default=None, help="one stem; else all of 5/6/7")
    p.add_argument("--limit", type=int, default=None,
                   help="cap new rows per stem (for smoke testing)")
    a = p.parse_args()
    for s in ([a.stem] if a.stem else STEMS):
        run_stem(s, a.profile, limit=a.limit)


if __name__ == "__main__":
    main()
