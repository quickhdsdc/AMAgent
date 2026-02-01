import re
import json
from pathlib import Path
import matplotlib.pyplot as plt


def find_latest_checkpoint(model_run_dir: Path) -> Path:
    """Given .../Exp_*/<model_key>/ return latest checkpoint-* folder."""
    ckpts = []
    for p in model_run_dir.glob("checkpoint-*"):
        m = re.search(r"checkpoint-(\d+)$", p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-* found in: {model_run_dir}")
    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def load_train_loss_from_trainer_state(trainer_state_path: Path):
    """Return (steps, losses). Uses 'loss' entries in log_history."""
    state = json.loads(trainer_state_path.read_text())
    log_history = state.get("log_history", [])
    steps, losses = [], []
    for row in log_history:
        if "step" in row and "loss" in row:
            steps.append(row["step"])
            losses.append(row["loss"])
    if not steps:
        raise ValueError(f"No (step, loss) found in log_history: {trainer_state_path}")
    return steps, losses


def collect_runs(base_dir: Path, exp_prefix: str, model_key: str):
    """
    exp_prefix: 'Exp_ID_' or 'Exp_OOD_'
    Returns list of dicts: {exp_name, steps, losses}
    """
    runs = []
    for exp_dir in sorted(base_dir.glob(f"{exp_prefix}*")):
        model_dir = exp_dir / model_key
        if not model_dir.exists():
            continue

        latest_ckpt = find_latest_checkpoint(model_dir)
        trainer_state = latest_ckpt / "trainer_state.json"
        if not trainer_state.exists():
            continue

        steps, losses = load_train_loss_from_trainer_state(trainer_state)
        runs.append(
            {
                "exp_name": exp_dir.name, 
                "steps": steps,
                "losses": losses,
                "ckpt": latest_ckpt.name,
            }
        )
    return runs


def plot_runs(runs, title: str, out_path: Path):
    if not runs:
        print(f"[WARN] No runs found for: {title}")
        return

    plt.figure(figsize=(7.5, 4.5))
    for r in runs:
        plt.plot(r["steps"], r["losses"], label=f"{r['exp_name']} ({r['ckpt']})")

    plt.xlabel("Training step")
    plt.ylabel("Training loss")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"Saved: {out_path}")


def plot_id_ood_for_model(base_dir: str, model_key: str, out_dir: str = None):
    base_dir = Path(base_dir)
    out_dir = Path(out_dir) if out_dir else base_dir

    id_runs = collect_runs(base_dir, "Exp_ID_", model_key)
    ood_runs = collect_runs(base_dir, "Exp_OOD_", model_key)

    plot_runs(
        id_runs,
        title=f"{model_key} — In-distribution (ID) training loss",
        out_path=out_dir / f"loss_ID_{model_key}.png",
    )
    plot_runs(
        ood_runs,
        title=f"{model_key} — Out-of-distribution (OOD) training loss",
        out_path=out_dir / f"loss_OOD_{model_key}.png",
    )


BASE = "./results/AM"
MODEL_KEY = "Qwen2.5_sc_lora-r32-a32" 
plot_id_ood_for_model(BASE, MODEL_KEY, out_dir="./AM")
