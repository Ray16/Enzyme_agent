"""Run a shell command file on a Globus Compute endpoint.\n\nargv: endpoint_id cmdfile outfile [walltime_s] [user_endpoint_config_json]\n\nThe config is required by multi-user (template) endpoints -- a submit without\none comes back 422 ENDPOINT_NOT_ONLINE even when the endpoint is healthy.\nPolaris wants {"account": "FRAME-IDP"}; aurora-gpu forbids `account` and takes\n{"queue", "walltime", "nodes_per_block", "accelerators"} (accelerators = tiles,\n1-12). See scripts/RESTART_GC_ENDPOINT.sh for the full schema notes.\n"""
import sys, json
from globus_compute_sdk import Executor
from globus_compute_sdk.sdk.shell_function import ShellFunction

ep, cmdf, outp = sys.argv[1], sys.argv[2], sys.argv[3]
wt = int(sys.argv[4]) if len(sys.argv) > 4 else 1800
cfg = json.loads(sys.argv[5]) if len(sys.argv) > 5 else None
# ShellFunction runs str.format() on the command, so every literal brace must
# be doubled or a bash `${VAR:-x}` / `{ ...; }` raises KeyError.
cmd = open(cmdf).read().replace('{', '{{').replace('}', '}}')
kw = {"user_endpoint_config": cfg} if cfg else {}
with Executor(endpoint_id=ep, **kw) as ex:
    r = ex.submit(ShellFunction(cmd, walltime=wt)).result(timeout=wt + 600)
open(outp, 'w').write(f"rc={r.returncode}\n{r.stdout}\n--STDERR--\n{(r.stderr or '')[-3000:]}\n")
print(f"rc={r.returncode}")
print(r.stdout[-6000:])
