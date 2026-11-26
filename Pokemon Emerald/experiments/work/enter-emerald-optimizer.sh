# Sourced by local WSL training launchers. File descriptor 7 holds the optimizer slot.
exec 7>"$official/controls/large-optimizer.lock"
required_kib=4000000
if [[ -s "$official/controls/file-backed-replay/verified.json" ]]; then
  export EMERALD_REPLAY_MMAP_DIR=/opt/emerald-rl/replay-cache
  required_kib=2950000
fi
while :; do
  if (( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) >= required_kib )); then
    flock 7
    if (( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) >= required_kib )); then break; fi
    flock -u 7
  fi
  sleep 60
done
