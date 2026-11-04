#!/bin/bash
# Restart the Globus Compute user endpoint on a Polaris login node.
# Target UUID: 4d01f9b2-5d45-46eb-8bc8-840733c4253f  (endpoint name: "polaris")
#
# The endpoint is a self-daemonizing process on ONE login node. It dies on login-node
# reboot / maintenance, and `globus-compute-endpoint list` reads shared-home state, so
# it can report an endpoint that is no longer actually running anywhere.
#
# ---------------------------------------------------------------------------------
# BEFORE YOU RUN THIS: a 422 ENDPOINT_NOT_ONLINE on submit usually does NOT mean the
# endpoint is dead.  "polaris" is a MULTI-USER (template) endpoint, so every submit
# must carry a user_endpoint_config; without one the manager cannot start a user
# endpoint and the API reports exactly that 422.  Verified working 2026-10-10:
#
#     Executor(endpoint_id=EP, user_endpoint_config={"account": "FRAME-IDP"})
#
# That bare config renders the LocalProvider template and runs on the login node.
# Adding queue/walltime/nodes_per_block renders a PBS block instead, and then the
# SAME 422 means "the PBS job has not started yet" -- a queue wait, not an error.
# The allowed keys are validated against ~/.globus_compute/polaris/user_config_schema.json
# on Polaris (additionalProperties:false): account, queue (debug|debug-scaling|
# preemptable|prod|demand), walltime HH:MM:SS, nodes_per_block, max_blocks,
# accelerators (1-4), idle_heartbeats_soft/hard, scheduler_options, worker_init.
#
# NOTE (2026-10-10): the FRAME-IDP Polaris allocation is OVERDRAWN
# (-1,088.4 node hours), so PBS-backed configs are not usable; login-node
# LocalProvider execution still works.
# ---------------------------------------------------------------------------------
ssh polaris 'bash -s' <<'SH'
set -u
echo "===== login node ====="; hostname

echo; echo "===== locate globus-compute-endpoint ====="
# A binary being executable is NOT enough: env/esmfold2/bin/globus-compute-endpoint has a
# broken shebang (points at /grand/FRAME-IDP/rzhu/envs/... -- missing "projects", and
# "envs" vs "env") and dies with "bad interpreter". It also sorts BEFORE gce in the glob,
# so the old first-match loop always picked the broken one. Probe each candidate by
# actually RUNNING it, and try the env the live daemons use (gce) first.
#
# The probe MUST use the `version` SUBCOMMAND. `--version` is not a flag this CLI
# accepts: it exits rc=2 even on a perfectly good binary (confirmed on 4.15.0), so a
# `--version` probe rejects every candidate and the script reports "NO WORKING
# globus-compute-endpoint FOUND" while the live daemons are happily running that very
# binary. That false negative is what this probe exists to avoid.
works() { [ -x "$1" ] && "$1" version >/dev/null 2>&1; }
GCE=""
for c in /grand/projects/FRAME-IDP/rzhu/env/gce/bin/globus-compute-endpoint \
         "$(command -v globus-compute-endpoint 2>/dev/null)" \
         /grand/projects/FRAME-IDP/rzhu/env/*/bin/globus-compute-endpoint \
         ~/.conda/envs/*/bin/globus-compute-endpoint \
         ~/anaconda3/envs/*/bin/globus-compute-endpoint \
         ~/anaconda3/bin/globus-compute-endpoint; do
  [ -n "${c:-}" ] || continue
  if works "$c"; then GCE="$c"; break; fi
  [ -x "${c:-}" ] && echo "  skip (broken): $c"
done
[ -z "${GCE:-}" ] && { echo "NO WORKING globus-compute-endpoint FOUND"; exit 1; }
echo "using: $GCE  ($("$GCE" version 2>&1 | head -1))"

echo; echo "===== is a daemon already alive on THIS node? ====="
pgrep -af "globus_compute_endpoint|globus-compute-endpoint" | head || echo "(none here)"

echo; echo "===== registered endpoints ====="
$GCE list

echo; echo "===== restart (only if needed) ====="
# Don't tear down a healthy daemon. The template endpoint we care about is the process
# running `... globus-compute-endpoint start polaris` (the uep.* children are per-user
# endpoints it spawns on demand and they come and go on their own).
#
# Check the REGISTERED status first, not just a local pgrep. ssh round-robins across
# login nodes, so the daemon is routinely on a node other than the one we landed on
# (seen 2026-10-10: daemon + uep.* children on login-04, this script on login-01).
# A pgrep-only test calls that "not running here" and restarts a perfectly healthy
# endpoint, which kills any in-flight user endpoints.
registered_running() { $GCE list 2>/dev/null | grep -E "polaris" | grep -qi "Running"; }

if registered_running || pgrep -af "globus-compute-endpoint start polaris" >/dev/null 2>&1; then
  echo "template endpoint 'polaris' reports RUNNING (registered status and/or local pgrep)."
  echo "A 422 ENDPOINT_NOT_ONLINE on submit is then almost certainly a missing"
  echo "user_endpoint_config, not a dead endpoint -- see the header of this script."
  echo "(pass FORCE=1 to restart anyway)"
  if [ "${FORCE:-0}" = "1" ]; then
    echo "FORCE=1 -> restarting"
    $GCE stop polaris 2>/dev/null || true; sleep 2; $GCE start polaris; sleep 8
  fi
else
  echo "not running here -- starting"
  $GCE stop polaris 2>/dev/null || true   # clears a stale pidfile from a rebooted node
  sleep 2
  $GCE start polaris
  sleep 8
fi

echo; echo "===== after ====="
$GCE list
pgrep -af "globus-compute-endpoint start polaris" | head -3
SH
