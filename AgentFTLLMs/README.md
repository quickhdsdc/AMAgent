# AgentFTLLMs

GPU-side fine-tuning and zero-shot inference for the AM defect-classification
study (companion to the AgentExperiments analysis repo). All paper experiments labelled
**C4 – C8, C10, C11** are produced here; results land under `results_AM/` and are
copied into the analysis repo for figure/table assembly.

---

## 1. Layout

```
AgentFTLLMs/
├── README.md                # this file
├── pyproject.toml / uv.lock # uv-managed env (Python 3.12)
├── run.sh                   # generic slurm batch entry point
├── start_vllm.sh            # vLLM OpenAI-compatible server launcher
├── scripts/
│   ├── run_C4C6_clm_4class.py    # C4 + C6  — 4-class CLM fine-tune
│   ├── run_C5C7_sc_4class.py     # C5 + C7  — 4-class SC fine-tune
│   ├── run_C4C6_clm_binary.py    # C4 + C6  — binary CLM fine-tune
│   ├── run_C5C7C10C11_sc.py      # C5/C7/C10/C11 SC ± physics, 4-class & binary
│   ├── run_C8_zeroshot_qwen3.py  # C8       — Qwen3-30B-Thinking zero-shot
│   └── utils/                    # clean_safetensor, plot_loss, verify_prompt
├── tasks/AM_defect_classification/   # internal modules (data, model, train, eval)
├── data/_am_qc/data_exp/             # train/test CSVs per stem
├── results_AM/                       # all run outputs — checkpoints, per-stem metrics, zero-shot CSVs
└── logs/                             # slurm stdout from each job
```

`scripts/run_*.py` selects the specific method via the `AM_METHOD` env var
where one runner covers multiple C labels — see [§4](#4-running-experiments).

---

## 2. Environment

* **Python:** 3.12 (pinned in `.python-version`)
* **Package manager:** `uv` — venv lives at `.venv/`
* **CUDA:** loaded by `run.sh` via `module load cuda/12.4`
* **HF cache:** set `HF_HOME` to a cache containing the required base models (offline mode).

One-time setup on a login node:

```bash
cd AgentFTLLMs
uv sync
# torch is intentionally not in pyproject — install with the CUDA wheel:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Edit `pyproject.toml` then re-run `uv sync` to refresh dependencies.

### Offline mode

`run.sh` exports `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`HF_DATASETS_OFFLINE=1`. All base models must already be in the local HF cache;
runner scripts use model IDs by default and allow local paths through
`AM_LLAMA_31_INSTRUCT_PATH`, `AM_QWEN_25_INSTRUCT_PATH`,
`AM_QWEN_25_BASE_PATH`, and `AM_QWEN3_THINKING_PATH`.

---

## 3. Slurm cluster

### Available GPU types

| GPU memory | Qwen2.5-14B QLoRA | Qwen3-30B 4-bit |
|------------|-------------------|-----------------|
| 40 GB      | Supported         | Insufficient    |
| 80 GB      | Supported         | Supported       |

> Select a GPU with enough memory for the requested model.

```bash
sinfo -o "%P %N %G %m"                 # list partitions and GRES
sinfo -N -o "%N %G %f" | grep gpu_mem  # see per-node memory tier
```

### Common Slurm commands

```bash
sbatch run.sh                          # submit default job (see §4)
squeue -u $USER                        # running / pending
scancel <JOBID>
sacct -u $USER --format=JobID,JobName,State,Elapsed,NodeList,AllocTRES%60 -X
tail -f AMT_<JOBID>.log                 # live stdout

# Interactive (debugging, with GPU):
srun --mem=80GB --cpus-per-task=2 --gres=gpu:a180:1 --pty bash
```

### run.sh — defaults and overrides

`run.sh` requests `--gres=gpu:a100:1 --mem=80GB --time=36:00:00` and runs
`scripts/${EXPERIMENT}.py`. Override per submission:

```bash
# C7 4-class SC (Qwen2.5-14B), default GPU
sbatch --export=ALL,EXPERIMENT=run_C5C7_sc_4class run.sh

# C10 4-class physics-informed input features
sbatch --export=ALL,EXPERIMENT=run_C5C7C10C11_sc,AM_METHOD=C10 run.sh

# C4 binary CLM (Llama-3.1-8B)
sbatch --export=ALL,EXPERIMENT=run_C4C6_clm_binary,AM_METHOD=C4 run.sh

# C8 zero-shot — needs the 80 GB GPU (a180), custom log
sbatch --gres=gpu:a180:1 \
       --output=logs/AMT_log_C8.txt \
       --export=ALL,EXPERIMENT=run_C8_zeroshot_qwen3 run.sh
```

`run.sh` exports `PYTHONPATH=$PROJECT_DIR` so the runner can import the
`tasks.AM_defect_classification.*` modules.

---

## 4. Running experiments

Mapping of method label → runner → required env vars:

| Method | Task | Runner | env |
|---|---|---|---|
| C4 | 4-class | `run_C4C6_clm_4class.py` | `AM_METHOD=C4` (default) |
| C6 | 4-class | `run_C4C6_clm_4class.py` | `AM_METHOD=C6` |
| C5 | 4-class | `run_C5C7_sc_4class.py` | edit `MODEL_KEY="llama3.1-8b"` |
| C7 | 4-class | `run_C5C7_sc_4class.py` | edit `MODEL_KEY="Qwen2.5-14B"` |
| C10 | 4-class | `run_C5C7C10C11_sc.py` | `AM_METHOD=C10`, `TASK=defect4` |
| C11 | 4-class | `run_C5C7C10C11_sc.py` | `AM_METHOD=C11`, `TASK=defect4` |
| C4 | binary | `run_C4C6_clm_binary.py` | `AM_METHOD=C4` |
| C6 | binary | `run_C4C6_clm_binary.py` | `AM_METHOD=C6` |
| C5 | binary | `run_C5C7C10C11_sc.py` | `AM_METHOD=C5`, `TASK=binary` |
| C7 | binary | `run_C5C7C10C11_sc.py` | `AM_METHOD=C7`, `TASK=binary` |
| C10 | binary | `run_C5C7C10C11_sc.py` | `AM_METHOD=C10`, `TASK=binary` |
| C11 | binary | `run_C5C7C10C11_sc.py` | `AM_METHOD=C11`, `TASK=binary` |
| C8 | binary | `run_C8_zeroshot_qwen3.py` | (none — Qwen3 30B 4-bit local) |

Each method writes to its own `results_AM/AM_<C>_*/` subdirectory; runs never
overwrite each other. Per-stem outputs:
* `Qwen2.5_sc_*_lora-r32-a32/eval_metrics_*.json`
* `Qwen2.5_sc_*_lora-r32-a32/eval_pred_*.csv`
* `all_experiments_summary_*.csv` (aggregated across stems)

Resume-friendly: existing checkpoints in `<out_dir>/checkpoint-*` are reused
unless `RESET_DIR=True` is set in the runner.

### C10 / C11 — physics-informed workflow

C10 = **PIF** (physics-informed features appended to the prompt, native CE
loss). C11 = **PIL** (VED-forbidden penalty loss, `L_total = L_CE + λ·L_phys`,
no physics features). Two-step procedure:

1. **λ-sweep** on the pilot stem `Exp_OOD_1` with `λ ∈ {0.1, 0.5, 1.0, 2.0}`:
   in the runner set `RUN_MODE="lambda_sweep"`, then `sbatch run.sh`.
   Summary → `results_AM/AM_C10_physics/lambda_sweep/_summary.csv`.
2. **Production**: pick λ\* (highest macro-F1, tiebreak on lowest
   `physics_violation_rate`), set `RUN_MODE="production"`,
   `LAMBDA_PHYS_PRODUCTION=<λ*>`, `RESET_DIR=True`, then `sbatch run.sh`.

### Result hand-off

After a run completes, copy the per-stem prediction CSVs and the summary
into the analysis repo:
```
AgentFTLLMs/results_AM/AM_<C>_*/Exp_*/.../eval_pred_*.csv
AgentFTLLMs/results_AM/AM_<C>_*/all_experiments_summary_*.csv
   →  AgentExperiments/results_AM/<C>_*/
```

---

## 5. vLLM serving (OpenAI-compatible endpoint)

`start_vllm.sh` launches an open-source LLM as an OpenAI-compatible HTTP
server, so AgentExperiments supervisor/sub-agent loops can call a locally hosted
model instead of Azure GPT-5. Use this when the call pattern is interactive
(KS → fusion → supervisor sequence). For one-shot batched inference on a
fixed prompt set, prefer the in-process `from vllm import LLM` route — but
in this repo we only need C8 zero-shot, which already uses transformers
directly (`run_C8_zeroshot_qwen3.py`).

### Server side (on the GPU node)

```bash
cd AgentFTLLMs

# Off-the-shelf Qwen3-30B on 4 GPUs
bash start_vllm.sh -m Qwen/Qwen3-30B-A3B-Thinking-2507 -t 4

# Qwen2.5-14B on 2 GPUs
bash start_vllm.sh -m Qwen/Qwen2.5-14B-Instruct -t 2

# Fine-tuned local model
bash start_vllm.sh -m ./results_AM/AM_C7_sc_binary/Exp_OOD_5/.../merged -t 2
```

Flags: `-m model`, `-p port` (default 8011), `-t TP_size` (default 4),
`-g gpu_memory_util` (default 0.90), `-h host` (default 127.0.0.1),
`-e "extra vllm args"`. Tool-call and reasoning parsers are auto-selected
for Qwen3 / Qwen2.5 / Llama families.

If `vllm` is not in the venv: `uv pip install 'vllm>=0.6.3' 'xgrammar>=0.1.11'`.

### Client side (your laptop)

```bash
ssh -L 8011:127.0.0.1:8011 <user>@<dgx-or-slurm-node>
curl http://127.0.0.1:8011/v1/models               # sanity check
```

Then in `AgentExperiments/config/config.toml`, pick the matching profile
(`qwen3_local`, `qwen25_local`, `llama_local`, or `vllm_remote` — edit
its `model` field to whatever you served) and run AgentExperiments against it.

### Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Connection refused` | SSH tunnel down, or vLLM not yet warm (2–5 min first load for 30B) |
| vLLM dies during model load | TP size > available GPUs; try lower `-t` |
| Tool-call parsing errors | Family parser mismatch — `-e '--tool-call-parser hermes'` |
| OOM at start | Lower `-g 0.85`, smaller model, or move to a180/h100 |
| Slow first request, then fast | Normal JIT warm-up (~30 s for 30B reasoning) |

---

## 6. Smoke-test the pipeline

After cloning / pulling, validate the venv + imports:

```bash
cd AgentFTLLMs
PYTHONPATH=. .venv/bin/python -c "
from tasks.AM_defect_classification.task_data_loader import TaskDataLoader
from tasks.AM_defect_classification.model_loader_sc import ModelLoader
from tasks.AM_defect_classification.evaluater_sc_physics import EvaluatorSeqClsPhysics
import scripts.run_C5C7C10C11_sc as _   # imports config block, does not train
print('OK')
"
```

If that prints `OK`, slurm submission via `run.sh` will resolve all imports.
