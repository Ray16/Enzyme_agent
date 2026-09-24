#!/usr/bin/env python3
"""Submit the 6 regioselectivity tetramer folds to Polaris via the Globus Compute
endpoint (login node writes a PBS script and qsubs it). Run from lambda with the
palm env (has globus_compute_sdk). Boltz-cp on A100, full tetramer, 10 samples,
--max_parallel_samples 1 (grand CLAUDE.md rule)."""
from globus_compute_sdk import Executor

EP = "4d01f9b2-5d45-46eb-8bc8-840733c4253f"                  # polaris login endpoint
BASE = "/grand/projects/FRAME-IDP/rzhu/Project/enzyme_agent/regioselectivity_repro"


def submit_job(base):
    import subprocess, os, textwrap
    os.makedirs(f"{base}/results", exist_ok=True); os.makedirs(f"{base}/logs", exist_ok=True)
    pbs = f"{base}/run_polaris.pbs"
    script = textwrap.dedent(f"""\
        #!/bin/bash
        #PBS -A FRAME-IDP
        #PBS -q debug
        #PBS -l select=1:system=polaris
        #PBS -l walltime=01:00:00
        #PBS -l filesystems=home:grand
        #PBS -N regio_repro
        set -x
        cd {base}
        mkdir -p results logs
        source /home/ray16/anaconda3/etc/profile.d/conda.sh
        conda activate /grand/projects/FRAME-IDP/rzhu/env/boltz-cp
        export BOLTZ_CACHE=/grand/projects/FRAME-IDP/rzhu/.boltz
        export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
        export OMP_NUM_THREADS=8
        run_gpu() {{ local g=$1; shift; for n in "$@"; do
            echo "[gpu$g] $n START $(date +%T)"
            CUDA_VISIBLE_DEVICES=$g boltz predict inputs/$n.yaml --out_dir results/$n \\
                --diffusion_samples 10 --max_parallel_samples 1 --output_format pdb \\
                --accelerator gpu --devices 1 --num_workers 1 --override \\
                > logs/$n.log 2>&1
            echo "[gpu$g] $n EXIT rc=$? $(date +%T)"
        done; }}
        names=$(for y in inputs/*.yaml; do basename $y .yaml; done)
        i=0; g0=(); g1=(); g2=(); g3=()
        for n in $names; do case $((i%4)) in
            0) g0+=($n);; 1) g1+=($n);; 2) g2+=($n);; 3) g3+=($n);; esac; i=$((i+1)); done
        run_gpu 0 "${{g0[@]}}" & run_gpu 1 "${{g1[@]}}" & run_gpu 2 "${{g2[@]}}" & run_gpu 3 "${{g3[@]}}" &
        wait
        touch {base}/DONE
        echo "ALL DONE $(date)"
        """)
    open(pbs, "w").write(script)
    r = subprocess.run(["qsub", pbs], capture_output=True, text=True, cwd=base)
    return {"rc": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}


if __name__ == "__main__":
    with Executor(endpoint_id=EP) as ex:
        print(ex.submit(submit_job, BASE).result(timeout=180))
