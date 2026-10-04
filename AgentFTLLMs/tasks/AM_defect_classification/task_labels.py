import os

TASK = os.environ.get("AM_TASK", "defect4")

if TASK == "binary":
    LABEL_ORDER = ["good", "defective"]
elif TASK == "defect4":
    LABEL_ORDER = ["none", "lof", "balling", "keyhole"]
else:
    raise ValueError(f"AM_TASK must be 'defect4' or 'binary', got {TASK!r}")

NUM_LABELS = len(LABEL_ORDER)