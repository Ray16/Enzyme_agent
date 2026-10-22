#!/usr/bin/env bash
# autodrive.sh -- run the watchdog + rolling analysis to completion, DETACHED from any session.
#
# Why this exists: the interactive monitor expires every 30 min and needs re-arming. The fold
# workers are setsid-detached and survive on their own, but the WATCHDOG would not - so dead jobs
# would stop being recovered and freed GPUs would stop being picked up. This loop removes that
# dependency entirely so the campaign finishes whether or not anyone is watching.
#
# Exits when step1 == 58 and step2 == 90, or after MAX_HOURS.
set -o pipefail
cd "$(dirname "$0")/.."
MAX_HOURS=${MAX_HOURS:-8}
SLEEP=${SLEEP:-600}
LOG=fleet/logs/autodrive.log
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))

count_done () {
  ls -d cofolds/$1/results/*/boltz_results_*/predictions/*/ 2>/dev/null | while read -r d; do
    [ "$(ls "$d"*.pdb 2>/dev/null | wc -l)" -ge 50 ] && echo x
  done | wc -l
}

echo "[$(date '+%F %T')] autodrive start (max ${MAX_HOURS}h, every ${SLEEP}s)" >> "$LOG"
while [ "$(date +%s)" -lt "$deadline" ]; do
  bash fleet/watchdog.sh >> "$LOG" 2>&1 || true
  bash roll.sh            >> "$LOG" 2>&1 || true
  s1=$(count_done step1); s2=$(count_done step2)
  echo "[$(date '+%F %T')] step1 $s1/58 · step2 $s2/90" >> "$LOG"
  if [ "$s1" -ge 58 ] && [ "$s2" -ge 90 ]; then
    echo "[$(date '+%F %T')] ALL 148 FOLDS COMPLETE -- final analysis" >> "$LOG"
    bash roll.sh >> "$LOG" 2>&1 || true
    exit 0
  fi
  sleep "$SLEEP"
done
echo "[$(date '+%F %T')] autodrive hit the ${MAX_HOURS}h limit; step1 $(count_done step1)/58 step2 $(count_done step2)/90" >> "$LOG"
