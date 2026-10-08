#!/bin/bash
# Common part of every exp/*.sh: one function per recorded command.
#   run   <output.json> VAR=value ... -- script.py arguments...     runs in the foreground (evaluations)
#   runbg <output.json> VAR=value ... -- script.py arguments...     runs in the background, at most $JOBS at a time (trainings)
#   waitall, report
# Every command starts from a clean set of the simulator's environment variables (prefixes BUB_ PPO_ SPPO_ OFF_ CAP_ STRAT_ EVAL_ VAL_
# DG_ TD_ OMP_ MKL_), so nothing leaks in from the calling shell; it runs inside $OUT, writes <output>.log next to the output, and is
# skipped if the output is already there (FORCE=1 reruns). SMOKE=1 shrinks every command to an execution test (see _smoke).
REPRO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPRO/env.sh"
SRC="$REPRO/src"
[ -n "$SMOKE" ] && OUT="${OUT}_smoke"
FAILDIR="$OUT/.failed/$(basename "$0" .sh)"            # failure markers of THIS script (a command that later succeeds removes its marker)
mkdir -p "$FAILDIR" "$CACHE"
[ -e "$OUT/sim_inputs.json" ] || cp "$SRC/sim_inputs.json" "$OUT/"
_PFX='^(BUB_|PPO_|SPPO_|OFF_|CAP_|STRAT_|EVAL_|VAL_|DG_|TD_|OMP_|MKL_)'

_deps() {      # checkpoints a command reads: in $OUT, or taken from ckpt/ (the ones the stored results were made with)
  local e v ck ok=0
  for e in "$@"; do
    case "$e" in CAP_LEARNED=*|TD_CKPT=*|DG_CKPT=*|SPPO_INIT=*|PPO_INIT=*) ;; *) continue ;; esac
    for v in $(echo "${e#*=}" | tr ',' ' '); do
      ck="${v#*=}"; case "$ck" in *.pt) ;; *) continue ;; esac
      [ -f "$OUT/$ck" ] && continue
      if [ "$SHIPPED" = 1 ] && [ -z "$SMOKE" ] && [ -f "$REPRO/ckpt/$ck" ]; then cp "$REPRO/ckpt/$ck" "$OUT/" && echo "    (trained checkpoint taken from ckpt/: $ck)"
      else echo "    MISSING checkpoint $ck - run the training script that writes ${ck%_*}.json (see MANIFEST.md)"; ok=1; fi
    done
  done
  return $ok
}

_smoke() {     # an execution test, not a result: 2 traffic seeds, first rate / buffer / channel count only, one training iteration
  case "$1" in
    cap_strat.py) export CAP_SEEDS=2 BUB_LAMS="${BUB_LAMS%%,*}" STRAT_BUFS="${STRAT_BUFS%%,*}" BUB_MS="${BUB_MS%%,*}" ;;
    time_decide.py) export TD_SEEDS=1 ;;
    diag_levels.py) export DG_SEEDS=1 ;;
  esac
}

_one() {
  local out=$1; shift; local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local script=$1; shift; local args=("$@")
  if [ -s "$OUT/$out" ] && [ -z "$FORCE" ]; then echo "  skip  $out (already there)"; return 0; fi
  if [ -z "$NODEPS" ] && ! _deps "${envs[@]}"; then touch "$FAILDIR/$out"; return 1; fi
  if [ -n "$SMOKE" ]; then case "$script" in sppo_hold.py|hppo_hold.py|offpol_hold.py) args[${#args[@]}-1]=1 ;; esac; fi
  local t0=$(date +%s)
  ( cd "$OUT" || exit 1
    for v in $(compgen -e | grep -E "$_PFX"); do unset "$v"; done
    for e in "${envs[@]}"; do export "$e"; done
    export CAP_SCHED_CACHE="$CACHE"
    [ -n "$SMOKE" ] && _smoke "$script"
    exec $PY "$SRC/$script" "${args[@]}" > "${out%.json}.log" 2>&1 )
  local rc=$?
  if [ $rc -eq 0 ] && [ -s "$OUT/$out" ]; then rm -f "$FAILDIR/$out"; echo "  done  $out  ($(( $(date +%s) - t0 )) s)"
  else touch "$FAILDIR/$out"; echo "  FAIL  $out (exit $rc) - see $OUT/${out%.json}.log"; fi
  return $rc
}

run()   { _one "$@"; }
runbg() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done; _one "$@" & }
waitall() { wait; }
report() {
  wait
  local n; n=$(ls "$FAILDIR" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -gt 0 ]; then echo "$(basename "$0"): $n command(s) failed or are waiting for a checkpoint:"; ls "$FAILDIR" | sed 's/^/    /'; exit 1; fi
  echo "$(basename "$0"): all commands done -> $OUT"
}
