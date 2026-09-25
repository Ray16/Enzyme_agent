import os, traceback
# one A100 per rank + self-shard via Polaris PALS env (no mpi4py needed)
lr = int(os.environ.get("PALS_LOCAL_RANKID", "0"))
os.environ["CUDA_VISIBLE_DEVICES"] = str(lr)          # set before torch import
gr = int(os.environ.get("PALS_RANKID", "0"))
nr = int(os.environ.get("BOLTZ_NRANKS", "1"))
from pathlib import Path
from boltz.main import predict
INP = os.environ["BOLTZ_INPUTDIR"]; OUT = os.environ["BOLTZ_OUTPUTDIR"]
CACHE = os.environ.get("BOLTZ_CACHE"); SAMPLES = int(os.environ.get("BOLTZ_TOTAL_SAMPLES", "50"))

def fold(y, out):
    predict.callback(data=str(y), out_dir=out, model="boltz2", accelerator="gpu",
        recycling_steps=3, sampling_steps=200, diffusion_samples=SAMPLES, use_msa_server=False,
        msa_server_url="https://api.colabfold.com", msa_pairing_strategy="greedy", output_format="pdb",
        cache=CACHE, checkpoint=None, affinity_checkpoint=None, devices=1, max_parallel_samples=1,
        step_scale=None, write_full_pae=False, write_full_pde=False, num_workers=2, override=False, seed=None,
        msa_server_username=None, msa_server_password=None, api_key_header=None, api_key_value=None,
        use_potentials=False, method=None, affinity_mw_correction=False, preprocessing_threads=1,
        max_msa_seqs=8192, subsample_msa=True, num_subsampled_msa=1024, no_kernels=False, write_embeddings=False)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    # longest-processing-first: sort by file size desc so big proteins spread across ranks and start early
    allc = sorted(Path(INP).glob("*.yaml"), key=lambda p: -p.stat().st_size)
    mine = allc[gr::nr]                                # this rank's shard
    for y in mine:
        o = os.path.join(OUT, y.stem)
        done = Path(o) / f"boltz_results_{y.stem}" / "predictions" / y.stem
        if done.is_dir() and len(list(done.glob("*_model_*.pdb"))) >= SAMPLES:
            print(f"[rank {gr}] skip done {y.stem}", flush=True); continue
        try:
            print(f"[rank {gr}] fold {y.stem}", flush=True)
            fold(y, o)
        except Exception:
            print(f"[rank {gr}] FAILED {y.stem}\n{traceback.format_exc()}", flush=True)
