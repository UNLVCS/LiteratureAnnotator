#!/usr/bin/env bash
# Full study run in waves: s00 on the 100 papers, then s01-s05 on the 25 resample papers.
#
#   ./study_v1/run/study_run.sh [s00|resamples|all]   (default: all)
#
# Each round collects finished batches, then submits whatever is missing. OpenRouter
# holds each in-flight batch's worst case against the balance, so submissions that do
# not fit are refused (402) and retried after the next collect. Papers that already
# have results are skipped, so the script can be stopped and rerun at any time.
# Resamples start only after every s00 paper is collected (BatchLabeler enforces this).
set -uo pipefail

cd "$(dirname "$0")/../.."
unset LA_CRITERIA  # always all six criteria
export LA_STUDY_CONFIG=config/study_models.yaml
export LA_MAX_BATCH_USD="${LA_MAX_BATCH_USD:-8}"
R=./study_v1/run/run.sh
# STUDY_MODELS="kimi" runs a subset; the default is every model in the study config.
read -r -a MODELS <<< "${STUDY_MODELS:-opus sol gemini deepseek qwen kimi}"
S00_PMIDS=study_v1/selection/study_v1_s00_pmids.txt
RESAMPLE_PMIDS=study_v1/selection/study_v1_resample_pmids.txt
POLL=${POLL:-300}
MAX_ROUNDS=${MAX_ROUNDS:-120}
phase=${1:-all}

quiet() { grep -vi "authorization\|bearer"; }

# One round for one sample: collect, then submit. Prints "OPEN" if work remains.
round() {
  local sample=$1 pmids=$2 out
  LA_SAMPLE=$sample LA_PMIDS_FILE=$pmids $R collect 2>&1 | quiet | grep -E "collecting|uploaded|\\$|not uploaded|Error|Traceback"
  out=$(LA_SAMPLE=$sample LA_PMIDS_FILE=$pmids $R submit "${MODELS[@]}" 2>&1 | quiet)
  echo "$out" | grep -E "submitted|refused|Error|Traceback|Collect sample|HTTP="
  local pending
  pending=$(LA_SAMPLE=$sample LA_PMIDS_FILE=$pmids $R status 2>&1 | grep -c "): ")
  if [ "$pending" -gt 0 ] || echo "$out" | grep -qE "[1-9][0-9]* to submit|refused|Collect sample|HTTP="; then
    echo "  s0$sample: $pending batch(es) pending"
    echo OPEN
  fi
}

run_phase() {
  local name=$1; shift
  local samples=("$@")
  for n in $(seq 1 "$MAX_ROUNDS"); do
    local open=0
    for k in "${samples[@]}"; do
      local pmids=$RESAMPLE_PMIDS
      [ "$k" -eq 0 ] && pmids=$S00_PMIDS
      round "$k" "$pmids" | tee /dev/stderr | grep -q "^OPEN$" && open=1
    done 2>&1
    echo "[$(date -u +%H:%M)] $name round $n: $([ $open -eq 1 ] && echo open || echo DONE)"
    [ $open -eq 0 ] && return 0
    sleep "$POLL"
  done
  echo "$name: gave up after $MAX_ROUNDS rounds"; return 1
}

case $phase in
  s00) run_phase s00 0 ;;
  resamples) run_phase resamples 1 2 3 4 5 ;;
  all) run_phase s00 0 && run_phase resamples 1 2 3 4 5 ;;
  *) echo "usage: $0 [s00|resamples|all]"; exit 2 ;;
esac
