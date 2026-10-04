import os
from pathlib import Path

TARGET_FILES = {
    "adapter_model.safetensors"
}

def cleanup_checkpoints(base_dir: str, model_key: str, dry_run: bool = True):
    """
    base_dir: path containing Exp_* folders (e.g., './AM')
    model_key: folder name under each experiment (e.g., 'llama3.1_sc_lora-r32-a32')
    dry_run: if True, only prints what would be deleted
    """
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Base dir not found: {base.resolve()}")

    total_candidates = 0
    total_deleted = 0
    total_bytes = 0

    exp_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("Exp_")])

    for exp in exp_dirs:
        model_dir = exp / model_key
        if not model_dir.exists():
            continue

        ckpt_dirs = sorted([p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")])
        for ckpt in ckpt_dirs:
            for fname in TARGET_FILES:
                fpath = ckpt / fname
                if fpath.exists() and fpath.is_file():
                    size = fpath.stat().st_size
                    total_candidates += 1
                    total_bytes += size

                    if dry_run:
                        print(f"[DRY RUN] Would delete: {fpath} ({size/1e6:.2f} MB)")
                    else:
                        try:
                            fpath.unlink()
                            total_deleted += 1
                            print(f"Deleted: {fpath} ({size/1e6:.2f} MB)")
                        except Exception as e:
                            print(f"[ERROR] Failed to delete {fpath}: {e}")

    print("\n=== Summary ===")
    if dry_run:
        print(f"Files matched for deletion: {total_candidates}")
        print(f"Estimated space to free: {total_bytes/1e9:.3f} GB")
        print("No files were deleted (dry_run=True).")
    else:
        print(f"Files deleted: {total_deleted}")
        print(f"Space freed: {total_bytes/1e9:.3f} GB")

BASE_DIR = "./results_AM/AM"
MODEL_KEY = "Qwen2.5_sc_lora-r32-a32"


cleanup_checkpoints(BASE_DIR, MODEL_KEY, dry_run=False)
