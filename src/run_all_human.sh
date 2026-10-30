#!/usr/bin/env bash
# Run the human pipeline (run_human.py, then fit_body_identity.py) on every Track 1 episode.
# Skips episodes whose fitted parameters already exist, keeps going past a failed episode, and
# writes one log per episode to work/human/logs/. Run from the project root inside v2d-ubuntu:
#
#     bash src/run_all_human.sh
set -u

PY="$HOME/v2d/.venv/bin/python"
export HF_TOKEN_PATH="$PWD/.secrets/hf_token.txt" HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p work/human/logs

echo "$(date '+%F %T') start"
for episode in $(seq 0 29); do
  name=$(printf "episode_%06d" "$episode")
  log="work/human/logs/$name.log"
  if [ -f "work/human/${name}_mhr.npz" ]; then
    echo "$(date +%T) $name already done"
    continue
  fi
  echo "$(date +%T) $name start"
  if "$PY" src/run_human.py --episode "$episode" > "$log" 2>&1 \
     && "$PY" src/fit_body_identity.py --episode "$episode" >> "$log" 2>&1; then
    echo "$(date +%T) $name done: $(grep -h -E 'seconds per frame|acceleration|identity fit' "$log" | tr '\n' ' ')"
  else
    echo "$(date +%T) $name FAILED, see $log"
  fi
done
echo "$(date '+%F %T') all episodes attempted"
