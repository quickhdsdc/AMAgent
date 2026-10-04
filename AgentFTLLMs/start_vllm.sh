#!/usr/bin/env bash
# start_vllm.sh — host an open-source LLM on the DGX for AM-Agent experiments
#
# Quick start (from the DGX, inside the AMTrainEval/ venv):
#   bash start_vllm.sh                                          # default Qwen3-30B-A3B-Thinking
#   bash start_vllm.sh -m Qwen/Qwen2.5-14B-Instruct -t 2         # Qwen2.5-14B on 2 GPUs
#   bash start_vllm.sh -m meta-llama/Meta-Llama-3.1-8B-Instruct -t 1
#   bash start_vllm.sh -m ./results/Qwen2.5-14B-LoRA-merged -t 2 # local fine-tuned model
#
# The server stays running. From your laptop:
#   ssh -L 8011:127.0.0.1:8011 <user>@dgx-host
# then in AMagent point the supervisor at the `qwen_local` profile in config.toml.

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────
MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507"
PORT=8011
TP_SIZE=4
GPU_MEM=0.90
HOST="127.0.0.1"
EXTRA=""

# ─── Parse arguments ────────────────────────────────────────────────
while getopts "m:p:t:g:h:e:" opt; do
    case $opt in
        m) MODEL="$OPTARG" ;;
        p) PORT="$OPTARG" ;;
        t) TP_SIZE="$OPTARG" ;;
        g) GPU_MEM="$OPTARG" ;;
        h) HOST="$OPTARG" ;;
        e) EXTRA="$OPTARG" ;;
        *) echo "Usage: $0 [-m model] [-p port] [-t tp_size] [-g gpu_mem] [-h host] [-e 'extra vllm args']"; exit 1 ;;
    esac
done

# ─── Resolve paths ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_BIN="${SCRIPT_DIR}/.venv/bin"
VLLM_BIN="${VENV_BIN}/vllm"

# Fall back to system vllm if no project venv
if [[ ! -x "${VLLM_BIN}" ]]; then
    if command -v vllm >/dev/null 2>&1; then
        VLLM_BIN="$(command -v vllm)"
        echo "[info] using system vllm at ${VLLM_BIN}"
    else
        echo "ERROR: vllm not found. Install via:" >&2
        echo "  uv pip install 'vllm>=0.6.3' 'xgrammar>=0.1.11'" >&2
        echo "  or  pip install vllm" >&2
        exit 1
    fi
fi

# ─── Auto-detect tool-call + reasoning parsers ──────────────────────
# Needed if the agent uses tool-calls or reasoning-content
TOOL_CALL_PARSER=""
REASONING_PARSER=""
PARSER_ARGS=""
LOWER_MODEL="$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')"
if [[ "${LOWER_MODEL}" == *qwen3* || "${LOWER_MODEL}" == *qwen-3* ]]; then
    TOOL_CALL_PARSER="qwen3_xml"
    REASONING_PARSER="qwen3"
elif [[ "${LOWER_MODEL}" == *qwen2.5* || "${LOWER_MODEL}" == *qwen-2.5* ]]; then
    TOOL_CALL_PARSER="hermes"
elif [[ "${LOWER_MODEL}" == *llama* ]]; then
    TOOL_CALL_PARSER="llama3_json"
fi
if [[ -n "${TOOL_CALL_PARSER}" ]]; then
    PARSER_ARGS="--enable-auto-tool-choice --tool-call-parser ${TOOL_CALL_PARSER}"
fi
if [[ -n "${REASONING_PARSER}" ]]; then
    PARSER_ARGS="${PARSER_ARGS} --reasoning-parser ${REASONING_PARSER}"
fi

# ─── Banner ─────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   vLLM Server Launcher — AMTrainEval / AM-Agent          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo "  Model    : ${MODEL}"
echo "  Endpoint : http://${HOST}:${PORT}/v1"
echo "  TP size  : ${TP_SIZE}"
echo "  GPU mem  : ${GPU_MEM}"
[[ -n "${TOOL_CALL_PARSER}" ]] && echo "  Tool parser    : ${TOOL_CALL_PARSER}"
[[ -n "${REASONING_PARSER}" ]] && echo "  Reasoning      : ${REASONING_PARSER}"
[[ -n "${EXTRA}" ]] && echo "  Extra args     : ${EXTRA}"
echo ""

# ─── Launch ─────────────────────────────────────────────────────────
exec "${VLLM_BIN}" serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${MODEL}" \
    --api-key "EMPTY" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEM}" \
    --trust-remote-code \
    ${PARSER_ARGS} \
    ${EXTRA}
