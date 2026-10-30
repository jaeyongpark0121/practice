#!/usr/bin/env bash
# Run the object stage (run_object.py), then merge it with the body fit (build_episode.py) and
# draw a check sheet (render_check.py), on every Track 1 episode. Skips episodes whose object
# track already exists (the merge is cheap and always reruns), keeps going past a failed episode,
# and logs each episode to work/object/logs/. Run from the project root inside v2d-ubuntu:
#
#     bash src/run_all_object.sh
set -u

PY="$HOME/v2d/.venv-obj/bin/python"
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p work/object/logs

echo "$(date '+%F %T') start"
for episode in $(seq 0 29); do
  name=$(printf "episode_%06d" "$episode")
  log="work/object/logs/$name.log"
  if [ -f "work/object/$name.npz" ]; then
    echo "$(date +%T) $name track exists" > "$log"
  elif ! "$PY" src/run_object.py --episode "$episode" > "$log" 2>&1; then
    echo "$(date +%T) $name FAILED in run_object, see $log"
    continue
  fi
  if "$PY" src/build_episode.py --episode "$episode" >> "$log" 2>&1 \
     && "$PY" src/render_check.py --episode "$episode" >> "$log" 2>&1; then
    echo "$(date +%T) $name done: $(grep -h -E 'object found|mask on|moved to hand|acceleration at' "$log" | tr '\n' ' ')"
  else
    echo "$(date +%T) $name FAILED in build, see $log"
  fi
done
echo "$(date '+%F %T') all episodes attempted"
