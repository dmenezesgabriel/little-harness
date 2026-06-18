#!/usr/bin/env bash
# Canonical CPU benchmark sweep behind SMALL_LLM_TUNING_REPORT.md.
#
# Run from the repo root so the file/search tools resolve against real files.
# Runs are sequential (one model loaded at a time) to avoid oversubscribing the
# 4 physical cores. Dense models (Phi-3 3.8B, qwen-7B) are slow on CPU and prone
# to multi-minute runaway generations, so they skip the expensive `all` tool set
# and are timeout-guarded; the fast MoE 8B-A1B models get full coverage.
#
#   bash packages/little-harness/evaluation/run_sweep.sh
set -u
ROOT=/home/gabriel-menezes/Documents/repos/local-llm
cd "$ROOT" || exit 1
export PYTHONPATH=packages/little-harness
BENCH=".venv/bin/python -m evaluation.benchmark"
RES=packages/little-harness/evaluation/results
mkdir -p "$RES"

base() { $BENCH --threads 4 --batch 256 --n-ctx 4096 --temperatures 0.1 \
  --max-iterations 5 "$@" --out "$RES/sweep_models.jsonl"; }

echo "###### SWEEP A: cross-model quality + tool-set tax"
rm -f "$RES/sweep_models.jsonl"
# Fast models (350M, 1.2B) and MoE 8B-A1B: full single/curated/all coverage.
for m in LFM2.5-350M-Q4_K_M.gguf LFM2.5-1.2B-Instruct-Q8_0.gguf \
         LFM2.5-8B-A1B-Q4_K_M.gguf LFM2-8B-A1B-Q4_K_M.gguf; do
  echo ">>> $(date +%H:%M:%S) $m"
  base --model "models/$m" --tool-sets single,curated,all
done
# Dense models: single/curated only, timeout-guarded (CPU runaway protection).
for m in Phi-3-mini-4k-instruct-q4.gguf qwen2.5-coder-7b-instruct-q4_k_m.gguf; do
  echo ">>> $(date +%H:%M:%S) $m (timeout 600s)"
  timeout 600 bash -c "$BENCH --threads 4 --batch 256 --n-ctx 4096 \
    --temperatures 0.1 --max-iterations 5 --model models/$m \
    --tool-sets single,curated --out $RES/sweep_models.jsonl" \
    || echo "($m timed out — partial)"
done

echo "###### SWEEP B: load-param grid on 1.2B (threads x batch x flash_attn)"
rm -f "$RES/sweep_loadparams.jsonl"
for th in 4 8; do for ba in 256 512; do for fa in "" "--flash-attn"; do
  echo ">>> $(date +%H:%M:%S) 1.2B threads=$th batch=$ba flash='$fa'"
  $BENCH --model models/LFM2.5-1.2B-Instruct-Q8_0.gguf --threads $th --batch $ba \
    --n-ctx 4096 $fa --temperatures 0.1 --max-iterations 5 --tool-sets single \
    --out "$RES/sweep_loadparams.jsonl"
done; done; done
echo "###### SWEEP B2: threads check on LFM2.5-8B-A1B"
for th in 4 8; do
  $BENCH --model models/LFM2.5-8B-A1B-Q4_K_M.gguf --threads $th --batch 256 \
    --n-ctx 4096 --temperatures 0.1 --max-iterations 5 --tool-sets single \
    --out "$RES/sweep_loadparams.jsonl"
done

echo "###### SWEEP C: runtime params on 1.2B (curated)"
rm -f "$RES/sweep_runtime.jsonl"
$BENCH --model models/LFM2.5-1.2B-Instruct-Q8_0.gguf --threads 4 --batch 256 \
  --n-ctx 4096 --temperatures 0.0,0.1,0.3 --max-iterations 5 --tool-sets curated \
  --out "$RES/sweep_runtime.jsonl"
$BENCH --model models/LFM2.5-1.2B-Instruct-Q8_0.gguf --threads 4 --batch 256 \
  --n-ctx 4096 --temperatures 0.1 --max-iterations 3,5,8 --tool-sets curated \
  --out "$RES/sweep_runtime.jsonl"

echo "###### SWEEP COMPLETE $(date +%H:%M:%S)"
echo "Summarize with: PYTHONPATH=packages/little-harness .venv/bin/python \\"
echo "  -m evaluation.summarize $RES/sweep_models.jsonl --group model,tool_set"
