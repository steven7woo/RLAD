# RLAD on SageMaker HyperPod — Setup & Troubleshooting Notes

Reference for getting `./RFT_pipeline.sh run` working on this cluster
(`/fsx/gstevenw/testing_alignment_algos/RLAD`). Written 2026-07-13.

The RLAD repo assumes a Slurm cluster with GPU gres and a working Pyxis/enroot
container stack. This HyperPod cluster differs on three points, each of which
blocked the pipeline in turn. All three are now resolved; this documents what
broke, why, and how it was fixed — so a future run (or a fresh clone) can
reproduce the fix quickly.

## Environment summary

- **Cluster:** AWS SageMaker HyperPod, Slurm 24.11.0
- **Partition:** `ml.p5.48xlarge`, 2 nodes: `ip-10-1-38-11`, `ip-10-1-81-8` (8× H100 80GB each)
- **Login/controller node:** `ip-10-1-97-143`
- **Conda env:** `rlad` at `/fsx/gstevenw/miniconda3` (built by pipeline bootstrap)
- **Container image:** `/fsx/gstevenw/testing_alignment_algos/images/miles.sqsh`
- **Lifecycle scripts:** `/fsx/gstevenw/ThemisHyperPodCluster/lifecycle_files`
- Passwordless `sudo` and passwordless `ssh` to compute nodes both work as `gstevenw`.

---

## Issue 1 — GPU allocation: no gres on this cluster

### Symptom
```
sbatch: error: Invalid generic resource (gres) specification
ERROR: failed to submit container_prep
```
Every job failed at submit time.

### Root cause
This cluster defines **no GPU gres**. `sinfo`/`scontrol` show `Gres=(null)`
and `CfgTRES=cpu=96,mem=2T,billing=96` (no `gpu`). Any `--gpus-per-node=N`
or `--gres=gpu:N` request is rejected. GPUs are obtained by allocating the
**whole node** with `--exclusive` (which exposes all 8 H100s).

Verify:
```bash
sinfo -p ml.p5.48xlarge -o '%N %G' -h          # -> "(null)"
sbatch --partition=ml.p5.48xlarge --gpus-per-node=1 --wrap='true'   # rejected
sbatch --partition=ml.p5.48xlarge --nodes=1 --exclusive --wrap='nvidia-smi -L'  # works, 8 GPUs
```

### Fix (committed: "Use exclusive whole-node allocation instead of GPU gres")
Replaced all gres-based GPU requests with `--exclusive`:

- 9 `#SBATCH --gpus-per-node=N` headers in `train/rl/jobs/*.sbatch`
  (`prepare_container`, `warmstart`, `rft_data`, `solgen_data`,
  `prep_megatron_ckpt`, `submit_train`, `eval`, `convert_hf`, `eval_rlad`)
  → `#SBATCH --exclusive`
- `srun --gpus-per-task=N` in `train/rl/jobs/run_gpu_shards.sh` and
  `train/rl/jobs/prepare_container.sbatch` → `--exclusive`
- `--gpus-per-node` in `train/rl/jobs/cluster_env.sh` (`rlad_inference_sbatch`)
  and `RFT_pipeline.sh` (`INFERENCE_SBATCH_ARGS`) → `--exclusive`

**Left unchanged (correct as-is):**
- `RLAD_GPUS_PER_NODE` internal shard math (`CUDA_VISIBLE_DEVICES=0..7`) — an
  exclusive node still exposes 8 GPUs, so the per-shard loop is right.
- Ray args `--actor-num-gpus-per-node 8` — these are application args, not Slurm flags.

---

## Issue 2 — `cluster_env.sh` not found inside batch jobs

### Symptom
```
/var/spool/slurmd/job00005/cluster_env.sh: No such file or directory
```
Job submitted fine but died immediately at runtime.

### Root cause
Every `.sbatch` sourced its env with:
```bash
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/cluster_env.sh"
```
Standard Slurm **copies the batch script into the node spool dir**
(`/var/spool/slurmd/job*/…`) before running it, so `${BASH_SOURCE[0]}`
resolves there — where `cluster_env.sh` does not exist. (This works on some
Slurm builds that execute the script in place, which is why upstream never hit it.)

### Fix (committed: "Anchor cluster_env.sh source to RLAD_HOME in batch scripts")
Anchored the source path to `$RLAD_HOME` (which the controller exports and
`--chdir`s into), with the old derivation kept as a fallback for manual host runs:
```bash
source "${RLAD_HOME:-$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/..}/jobs/cluster_env.sh"
```
Applied to all 9 `train/rl/jobs/*.sbatch` files. Host-side scripts
(`chain.sh`, `sbatch.sh`) were left alone — they run on the login node where
`BASH_SOURCE` resolves correctly.

### Note
`run`/`resume` refuse to start if the git tree is dirty
("tracked repository files are modified"). Both fixes above had to be
**committed** before the pipeline would proceed.

---

## Issue 3 — Pyxis/enroot container stack (the big one)

### Symptom
```
srun: unrecognized option '--container-image=...'
```
from the container-import job, even though `srun --help` on the login node
*did* list the pyxis options. A control test requesting an `alpine` image
silently ran on the **Ubuntu 22.04 host** instead of a container.

### Root cause
Pyxis was installed only on the **login node** (Jul 12); the two p5 compute
nodes were provisioned earlier (Jul 7) and never got it. `/opt/slurm` and
`/usr/local` are **node-local** (not shared), so:

| | Login node | Compute nodes |
|---|---|---|
| pyxis in `plugstack.conf` | yes | no (only `spank_auto_resume.so`) |
| `spank_pyxis.so` on disk | yes | absent |
| enroot installed | yes | absent |
| `slurmd` start | — | Jul 7 (predates pyxis) |

The login node advertised the options because it read a plugstack that
referenced the plugin; the compute-node `slurmd` had no plugin at all, so
`--container-image` fell through to the bare host.

### Fix — part A: install enroot+pyxis on the compute nodes
Used the cluster's **own official installer** (no custom hacks):
`lifecycle_files/utils/install_enroot_pyxis.sh`. It installs enroot debs,
builds pyxis 0.19.0 from source against `/opt/slurm/include`, wires
`plugstack.conf.d/pyxis.conf`, sets `ConstrainDevices=yes`, configures enroot
paths on `/opt/dlami/nvme`, and restarts `slurmd`.

Per node (both p5s):
```bash
ssh <node> '
  sudo apt-get -y install git build-essential          # node lacked git; pyxis build needs it
  cd /fsx/gstevenw/ThemisHyperPodCluster/lifecycle_files/utils
  sudo bash install_enroot_pyxis.sh compute
'
```
Verify the plugin loaded and containers actually isolate:
```bash
srun --partition=ml.p5.48xlarge --nodelist=<node> --nodes=1 --ntasks=1 --exclusive \
  --container-image=docker.io#alpine:latest cat /etc/os-release   # must print Alpine, not Ubuntu
```

**Persistence:** already handled — `lifecycle_files/config.py` has
`enable_docker_enroot_pyxis = True`, so any replaced/rebooted node auto-runs
the installer at provision time. No lifecycle edit was needed.

### Fix — part B: pre-import the Miles image (pyxis import is broken here)
After part A, **running** containers from a local `.sqsh` works, but
**importing** a registry image *through pyxis/slurmstepd* fails:
```
pyxis: [ERROR] Could not process JSON input
pyxis: curl: (23) Failure writing output to destination
```
This is a slurmstepd-environment quirk — a direct `enroot import` (or an
import inside a plain `srun` step) succeeds. Since RLAD only imports **once**
(in `prepare_container`, saving a local `miles.sqsh`) and every later job runs
from that local file, the workaround is to do the one import outside stepd.

Additional gotcha: the shared cache `/fsx/enroot` holds layer blobs owned by
other users with mode `-rw-r-----` (enroot keys layers by content hash and
tries to reuse them), causing `tar: … Permission denied`. Use a private cache.

```bash
IMG=/fsx/gstevenw/testing_alignment_algos/images/miles.sqsh
ssh ip-10-1-38-11 "
  mkdir -p /fsx/gstevenw/enroot_cache
  ENROOT_CACHE_PATH=/fsx/gstevenw/enroot_cache \
    enroot import -o ${IMG}.manualpartial 'docker://radixark/miles:dev-cu12-202606172131'
"
mv -f ${IMG}.manualpartial ${IMG}
# Write the provenance receipt the controller checks (training_container_ready):
{ printf 'schema=1\n'
  printf 'source=%s\n' 'docker.io#radixark/miles:dev-cu12-202606172131'
  printf 'bytes=%s\n' "$(stat -c '%s' ${IMG})"
  printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > ${IMG}.source
rm -rf /fsx/gstevenw/enroot_cache        # ~23 GB temp, safe to remove after import
```
The receipt must record `source=docker.io#radixark/miles:dev-cu12-202606172131`
(the Pyxis URI form) and a `bytes=` equal to the file size, or
`training_container_ready()` rejects it.

### Validation performed
- RLAD's own `scripts/verify_training_container.py` run **inside** the image
  via pyxis: passes — CUDA 12.9, H100 (capability 9.0), and
  apex/flash_attn/megatron/miles/ray/sglang/torch/transformer_engine all import.
- Nested `sbatch → srun --container-image=<local sqsh>` (how every training
  job actually launches): `cuda=True NVIDIA H100 80GB HBM3`.
- On `run`, the controller logs `Training container already ready` and skips
  the import stage entirely.

---

## Running / monitoring

```bash
# Launch (inside tmux; long-running, resumable):
tmux attach -t rlad-rft        # or: tmux new -s rlad-rft
./RFT_pipeline.sh run 2>&1 | tee -a pipeline_run.log

# Status without submitting anything:
./RFT_pipeline.sh status
squeue -u $USER
```

- Ctrl-C stops only the controller, not running Slurm jobs. Rerun `run`/`resume` to reconnect.
- Uses **both** p5s (16 shards) for inference/eval stages; single node for training.
- Publishes datasets+models to private HF (`rlad-original` prefix); logs 5
  DeepScaleR-hard metrics to W&B project `repro-paper003-rlad`.
- HF and W&B use cached logins (`hf auth login` / `wandb login`); the
  "HF_TOKEN/WANDB_API_KEY unset" preflight lines are harmless warnings.

## Quick recovery checklist (fresh clone or node replacement)

1. `git` history already contains the Issue 1 & 2 fixes — no re-editing needed if reusing this checkout.
2. If a p5 was replaced, confirm pyxis: `srun --nodelist=<node> --exclusive --container-image=docker.io#alpine:latest cat /etc/os-release` prints Alpine. If not, re-run `install_enroot_pyxis.sh compute` on it.
3. Confirm `images/miles.sqsh` + `images/miles.sqsh.source` exist and bytes match. If missing, redo Fix 3B.
4. `./RFT_pipeline.sh run`.
